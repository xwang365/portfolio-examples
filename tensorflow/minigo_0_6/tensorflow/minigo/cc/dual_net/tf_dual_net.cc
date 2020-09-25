// Copyright (c) 2020 Graphcore Ltd.
// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file has been modified by Graphcore Ltd.
// It has been modified to run the application on IPU hardware.

// Original file has been changed.
// It defines the major interface between the IPU and the TF C++ API.

#include "tf_dual_net.h"

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "../constants.h"
#include "../file/path.h"
#include "../logging.h"
#include "../thread_safe_queue.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

#if MINIGO_ENABLE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/stream_executor/platform.h"
#endif

// IPU configuration
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/core/framework/node_def_util.h"

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

#define MY_DEBUG 0

using namespace xla::poplarplugin;
namespace minigo {
namespace {

class TfDualNet : public DualNet {
  class TfWorker {
   public:
    TfWorker(Session* session) : batch_capacity_(0) {
      session_ = session;

      output_names_.emplace_back("policy_output");
      output_names_.emplace_back("value_output");
    }

    ~TfWorker() {
        if (session_ != nullptr) {
	    session_->Close();
	}
    };


    void RunMany(std::vector<const BoardFeatures*> features,
                 std::vector<Output*> outputs) {
      size_t num_features = features.size();
      //define the (maximum) batch size to avoid recompilation.
      #if MINIGO_SELFPLAY
        Reserve(1368);
      #endif
      #if MINIGO_EVAL
        Reserve(100*8/MY_DEVICE_COUNT/2);
      #endif
      // Memory reservation only needed for wrong configurations
      // when preconfigured batch size is not working
      Reserve(num_features);
      #if MY_DEBUG
      printf("Input size %d\n", inputs_[0].second.flat<float>().size());
      printf("Batch capacity: %d\n", batch_capacity_);
      printf("Input data size %d\n", inputs_[0].second.flat<float>().data().size());
      #endif
      auto* feature_data = inputs_[0].second.flat<float>().data();
      // Copy the features into the input tensor.
      for (const auto* feature : features) {
        feature_data =
            std::copy(feature->begin(), feature->end(), feature_data);
      }

      TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

      // Copy the policy and value out of the output tensors.
      const auto& policy_tensor = outputs_[0].flat<float>();
      const auto& value_tensor = outputs_[1].flat<float>();
#if MY_DEBUG
      printf("policy size %d\n\n", policy_tensor.size());
#endif
      for (size_t i = 0; i < num_features; ++i) {
        memcpy(outputs[i]->policy.data(), policy_tensor.data() + i * kNumMoves,
               sizeof(outputs[i]->policy));
        outputs[i]->value = value_tensor.data()[i];
      }
    }

   private:

    void Reserve(size_t capacity) {
      MG_CHECK(capacity > 0);
      if (capacity <= batch_capacity_) {
        return;
      }
      inputs_.clear();
      inputs_.emplace_back(
          "pos_tensor",
          Tensor(DT_FLOAT, TensorShape({static_cast<int>(capacity), kN, kN,
                                        kNumStoneFeatures})));
      batch_capacity_ = capacity;
    }

    Session *session_;
    std::vector<std::pair<std::string, Tensor>> inputs_;
    std::vector<std::string> output_names_;
    std::vector<Tensor> outputs_;
    size_t batch_capacity_;
  };

  struct InferenceData {
    std::vector<const DualNet::BoardFeatures*> features;
    std::vector<DualNet::Output*> outputs;
    absl::Notification* notification;
  };

 public:
  TfDualNet(std::string graph_path, int device_count);

  ~TfDualNet() override;

  void RunMany(std::vector<const BoardFeatures*> features,
               std::vector<Output*> outputs, std::string* model) override;

 private:
  static void PlaceOnDevice(GraphDef* graph_def, const std::string& device) {
    for (auto& node : *graph_def->mutable_node()) {
      AddNodeAttr("_XlaCompile", true, &node);
      AddNodeAttr("_XlaScope", "jit_socpe_ipu_0", &node);
      AddNodeAttr("_XlaSeparateCompiledGradients", false, &node);
      node.set_device(device);
      #if MY_DEBUG
//        node.PrintDebugString();
        std::cout << node.name() << std::endl;
      #endif
    }
  }

  std::string graph_path_;
  ThreadSafeQueue<InferenceData> inference_queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;
};

TfDualNet::TfDualNet(std::string graph_path, int device_count)
    : DualNet(std::string(file::Stem(graph_path))),
      graph_path_(graph_path),
      running_(true) {

  // global variable to avoid multiple device initializations in eval.cc
  // and to control device assignment for eval.cc
  static int num_calls = 0;

  GraphDef graph_def;

  //checkpoint meta file, including graphdef
  tensorflow::MetaGraphDef meta_graph_def;

  auto* env = Env::Default();

  const std::string export_dir = graph_path;

  // Platform needs to be reserved before loading model.
  // We set platform only once in case of double call for evaluation.
  // We keep track of it because the call numbers correspond to different
  // networks that need to be placed onto different IPUs.
  // There are two initial networks and two that are actually used at the end.
  if (num_calls == 0){
      GraphDef graph_def;
      SessionOptions options = SessionOptions();
      Session* session;
      session = NewSession(options);
      // determine parent parent directory for result files
      // this should contain prerecorded IPU configurations
      std::string str = graph_path;
      str = str.substr(0, str.find_last_of("/\\"));
      str = str.substr(0, str.find_last_of("/\\"));
      str = str.substr(0, str.find_last_of("/\\"));

      str = absl::StrCat(str, "/", MY_DEVICE_COUNT, "_ipu_init_graph_def.pb");

      ReadBinaryProto(env, str, &graph_def);

      session->Create(graph_def);

      session->Run({}, {}, {"IpuConfigureHardware"}, nullptr);
  }

  num_calls++;

  #if MY_DEBUG
    std::cout << "num of TfDualNet constructors: " << num_calls << std::endl;
  #endif

  // checkpoint loading approach
  TF_CHECK_OK(ReadBinaryProto(env, absl::StrCat(graph_path, ".meta"), &meta_graph_def));
  graph_def = meta_graph_def.graph_def();

  // Check that we're not loading a TPU model.
  for (const auto& node : graph_def.node()) {
    //MG_LOG(INFO) <<"node name: " << node.name();
    MG_CHECK(!absl::StartsWithIgnoreCase(node.name(), "tpu"))
        << "found node named \"" << node.name()
        << "\", this model looks like it was compiled for TPU";
  }

  auto session_functor = [this](Session* session) {
    TfWorker worker(session);
    while (running_) {
      InferenceData inference;
      if (inference_queue_.PopWithTimeout(&inference, absl::Milliseconds(10))) {
        worker.RunMany(std::move(inference.features),
                       std::move(inference.outputs));

        inference.notification->Notify();
      }
    }
  };

  // Create two worker threads per device (or two threads for CPU inference if
  // there are no accelerator devices present).
  // The number of threads created here must match the value returned by
  // TfDualNetFactory::GetBufferCount().

  // We differentiate between selfplay and evaluate because evaluate needs
  // a special treatment. For evaluate, there are four calls.
  // Two for initialization and two for the real execution.
  // They correspond to different networks (weight distributions).
  // Each network comes with its own program/weights and so putting them
  // onto different devices, avoids that the respective executables are changed.
  // This part is crucial for performance.

#if MINIGO_SELFPLAY
  for (int i = 0; i < std::max(device_count, 1); ++i){
#endif
#if MINIGO_EVAL
for (int i = (num_calls-1) * (device_count/2) ; i < (num_calls) * (device_count/2); ++i){
#endif

    int device = i;

    #if MINIGO_EVAL
      // The first two num_calls are for initialization purposes probably
      // and not really relevant
      // However, they have to be on different devices.
      if (num_calls<=2){
        device = i;
      } else {
        // For num_calls in {3, 4}, we need to correct the loop offset
        // to start at zero again.
        device = i -  device_count;
      }
    #endif

    // protobuf approach and checkpoint approach of placing graph on device in session
    PlaceOnDevice(&graph_def, absl::StrCat("/device:IPU:", device));
    SessionOptions options = SessionOptions();
    options.config.set_allow_soft_placement(true);
    options.config.set_log_device_placement(false);
    Session* session;
    session = NewSession(options);
    TF_CHECK_OK(session->Create(graph_def));

    // checkpoint variable initialization approach
    Tensor checkpointPathTensor(tensorflow::DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = graph_path;
    TF_CHECK_OK(session->Run(
      {{meta_graph_def.saver_def().filename_tensor_name(),
        checkpointPathTensor},},
      {},
      {meta_graph_def.saver_def().restore_op_name()},
      nullptr));

    // Two workers alternate on one IPU.
    // One processing while the other one is gathering and preparing data.
    worker_threads_.emplace_back(session_functor, session);
    worker_threads_.emplace_back(session_functor, session);
  }

}

TfDualNet::~TfDualNet() {
  running_ = false;
  for (auto& thread : worker_threads_) {
    thread.join();
  }
}

void TfDualNet::RunMany(std::vector<const BoardFeatures*> features,
                        std::vector<Output*> outputs, std::string* model) {
  MG_DCHECK(features.size() == outputs.size());

  absl::Notification notification;
  inference_queue_.Push(
      {std::move(features), std::move(outputs), &notification});
  notification.WaitForNotification();

  if (model != nullptr) {
    *model = graph_path_;
  }
}
}  // namespace

TfDualNetFactory::TfDualNetFactory() : device_count_(0) {
#if MINIGO_ENABLE_GPU
  if (tensorflow::ValidateGPUMachineManager().ok()) {
    device_count_ = tensorflow::GPUMachineManager()->VisibleDeviceCount();
  }
#endif
  device_count_ = MY_DEVICE_COUNT; // set device count here
}

int TfDualNetFactory::GetBufferCount() const {
  // The buffer count needs to match the number of worker_threads_ that
  // TfDualNet will create.
  return 2 * std::max(device_count_, 1);
}

std::unique_ptr<DualNet> TfDualNetFactory::NewDualNet(
    const std::string& model) {
  return absl::make_unique<TfDualNet>(model, device_count_);
}

}  // namespace minigo
