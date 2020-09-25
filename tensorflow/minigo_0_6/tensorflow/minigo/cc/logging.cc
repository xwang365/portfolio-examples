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

#include "logging.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "absl/synchronization/mutex.h"

namespace minigo {
namespace internal {

namespace {

absl::Mutex* mutex() {
  static auto* m = new absl::Mutex();
  return m;
}

void DumpStackTrace(std::ostream* os) {
  void* stack[64];
  int depth = absl::GetStackTrace(stack, 64, 1);
  char buffer[256];
  for (int i = 0; i < depth; ++i) {
    (*os) << "  " << stack[i] << "  ";
    if (absl::Symbolize(stack[i], buffer, 256)) {
      (*os) << buffer;
    } else {
      (*os) << "??";
    }
    (*os) << '\n';
  }
}

}  // namespace

LogStream::LogStream(const char* file, int line, LogLevel level)
    : level_(level) {
  if (level == LogLevel::INFO) {
    // We don't add a prefix to MG_LOG(INFO) log lines because many things rely
    // on the exact string being printed (GTP, correct formatting of position &
    // node descriptions, etc).
    return;
  }

  char c;
  switch (level) {
    case LogLevel::WARNING:
      c = 'W';
      break;
    case LogLevel::ERROR:
      c = 'E';
      break;
    case LogLevel::FATAL:
      c = 'F';
      break;
    default:
      c = 'U';
      break;
  }
  file = std::strrchr(file, '/');
  if (file == nullptr) {
    file = std::strrchr(file, '\\');
  }
  if (file == nullptr) {
    file = file;
  } else {
    file += 1;
  }
  *this << '[' << c << "] " << file << ':' << line << " : ";
}

LogStream::~LogStream() {
  {
    absl::MutexLock lock(mutex());
    std::cerr << stream_.rdbuf() << '\n';
    if (level_ == LogLevel::FATAL) {
      DumpStackTrace(&std::cerr);
    }
    std::cerr << std::flush;
  }
  if (level_ == LogLevel::FATAL) {
    exit(1);
  }
}

CheckFailStream::CheckFailStream(const char* cond, const char* file, int line)
    : impl_(file, line, LogLevel::FATAL) {
  impl_ << "check failed: " << cond << '\n';
}

}  // namespace internal
}  // namespace minigo
