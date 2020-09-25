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

#ifndef CC_MOVE_H_
#define CC_MOVE_H_

#include <string>

#include "color.h"
#include "coord.h"

namespace minigo {

struct Move {
  Move() = default;
  Move(Color color, Coord c) : color(color), c(c) {}

  Color color = Color::kEmpty;
  Coord c = Coord::kInvalid;

  std::string ToSgf() const;

  bool operator==(const Move& other) const {
    return color == other.color && c == other.c;
  }
};

}  // namespace minigo

#endif  // CC_MOVE_H_
