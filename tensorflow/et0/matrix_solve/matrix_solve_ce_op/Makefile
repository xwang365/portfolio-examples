# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


CC = g++

all: libmatrix_solve_ce_op.so

matrix_solve_codelets.gp: matrix_solve_codelets.cpp
	popc --target=ipu1,ipu2 -O2 matrix_solve_codelets.cpp -o $@

libmatrix_solve_ce_op.so: matrix_solve_codelets.gp main.cpp
	$(CC) -O3 main.cpp -shared -fpic -Wl,-soname,$@ -o $@ -lpoplar -lpoputil

clean:
	rm libmatrix_solve_ce_op.so matrix_solve_codelets.gp