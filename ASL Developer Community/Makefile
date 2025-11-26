# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: lint test install clean format

# ==============------------------------------------------------------------------------------------============== #
#                                                        Lint                                                       #
# ==============------------------------------------------------------------------------------------============== #
lint:
	poetry run pre-commit run --all-files --show-diff-on-failure

# ==============------------------------------------------------------------------------------------============== #
#                                                        Test                                                       #
# ==============------------------------------------------------------------------------------------============== #
test:
	poetry run coverage run -m pytest \
		--disable-warnings \
		--tb=short
	poetry run coverage report -m

# ==============------------------------------------------------------------------------------------============== #
#                                                       Install                                                     #
# ==============------------------------------------------------------------------------------------============== #
install:
	poetry install --with dev

# ==============------------------------------------------------------------------------------------============== #
#                                                        Format                                                     #
# ==============------------------------------------------------------------------------------------============== #
format:
	poetry run black src/ tests/ scripts/
	poetry run isort src/ tests/ scripts/
	poetry run autoflake --in-place --remove-all-unused-imports --remove-unused-variables --expand-star-imports --recursive src/ tests/ scripts/

# ==============------------------------------------------------------------------------------------============== #
#                                                        Clean                                                      #
# ==============------------------------------------------------------------------------------------============== #
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
