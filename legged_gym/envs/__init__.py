# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1 import A1
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1.flat.a1_flat_config import A1FlatCfg, A1FlatCfgPPO
from .go1.go1_config import Go1FlatCfg, Go1FlatCfgPPO
# from .go1.go1_rough_config import Go1RoughCfg, Go1RoughCfgPPO
from .go1.go1 import Go1
from .go2.go2 import Go2
from .go2.go2_config import GO2FlatCfg, GO2FlatCfgPPO
from .yuna.yuna_config import YunaRoughCfg, YunaRoughCfgPPO
from .yuna.yuna import Yuna
from .mit_humanoid.mit_humanoid import Humanoid
from .mit_humanoid.mit_humanoid_config import HumanoidCfg, HumanoidCfgPPO
from .yuna.mixed_terrains.yuna_rough_config import YunaActualRoughCfg, YunaActualRoughCfgPPO
from .h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from .h1.h1_env import H1Robot

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", A1, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1_flat", A1, A1FlatCfg(), A1FlatCfgPPO() )
task_registry.register( "go1_flat", Go1, Go1FlatCfg(), Go1FlatCfgPPO() )
# task_registry.register( "go1_rough", Go1, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register( "go2_flat", Go2, GO2FlatCfg(), GO2FlatCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "yuna", Yuna, YunaRoughCfg(), YunaRoughCfgPPO() )
task_registry.register( "yuna_rough", Yuna, YunaActualRoughCfg(), YunaActualRoughCfgPPO() )
task_registry.register("humanoid", Humanoid, HumanoidCfg(), HumanoidCfgPPO())
# task_registry.register( "yuna", LeggedRobot, YunaRoughCfg(), YunaRoughCfgPPO() )
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
