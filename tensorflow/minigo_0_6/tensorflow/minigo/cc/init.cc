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

#include "init.h"

#include "absl/debugging/symbolize.h"
#include "gflags/gflags.h"

namespace minigo {

void Init(int* pargc, char*** pargv) {
  gflags::SetUsageMessage((*pargv)[0]);
  absl::InitializeSymbolizer((*pargv)[0]);
  gflags::ParseCommandLineFlags(pargc, pargv, true);
}

}  // namespace minigo
