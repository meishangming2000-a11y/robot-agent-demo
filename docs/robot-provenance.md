# 机器人来源与替代边界

本记录说明公开模型的来源与实际适配。截至 2026-11-17，组合场景已完成本仓库记录的仿真验证及首个工具任务验证；测量与范围见[验证记录](validation.md)。尚未进行真机验证。

## 固定来源

上游为 Google DeepMind 的 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，固定提交：

`370c43965323c0944b433b202aac5058b5f907a9`（上游提交日期 2026-11-10）。

| 部件 | 固定文件 | 许可与说明 |
| --- | --- | --- |
| Panda 七轴机械臂，无原夹爪 | [panda_nohand.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/franka_emika_panda/panda_nohand.xml) | [Apache-2.0](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/franka_emika_panda/LICENSE)，[来源说明](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/franka_emika_panda/README.md) |
| Shadow E3M5 右手 | [right_hand.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/shadow_hand/right_hand.xml) | [Apache-2.0](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/shadow_hand/LICENSE)，Copyright 2022 Shadow Robot Company Ltd；[来源说明](https://github.com/google-deepmind/mujoco_menagerie/blob/370c43965323c0944b433b202aac5058b5f907a9/shadow_hand/README.md)明确原 URDF 与资产由 Shadow Robot Company 提供 |

两个模型目录在该提交均无单独 `NOTICE` 文件。再分发时保留各目录原 `LICENSE`、已有归属信息和固定提交；修改的文件需明确标记修改。项目自身的许可不覆盖或替换这些第三方许可。

最小原模型依赖为 Panda 的 59 个网格（32,349,018 字节）和 Shadow 右手的 13 个网格（3,382,030 字节），合计约 34.08 MiB，加 XML 与许可文本；无需下载整个模型库。上述数目包含原 Shadow 前臂网格，最终腕手组合可能不再引用其中部分文件。

## 本项目选择

采用 Panda `panda_nohand.xml` 与 Shadow 的 `rh_wrist` 以下子树，通过本项目定义的简单法兰连接。组合场景、法兰及控制器在独立项目内生成，不依赖同学四轴臂的代码、关节参数或几何，也不从原 Isaac 场景转换机器人。

原 Shadow 模型共有 25 个带质量的刚体，XML 质量相加为 **3.794 kg**，其中 `rh_forearm` 为 **3 kg**。本适配删除整个 `rh_forearm` 外壳、碰撞几何及其惯量，保留腕手子树 **0.794 kg**，另计自定义法兰质量。此变更是仿真模型适配，不代表真实 Shadow 产品可直接以该质量安装，也不构成实物法兰设计。

腕部 XYZ/姿态运动台可用于隔离手部接触问题；最终机械臂成果须以 Panda 的真实关节运动和动力学验证。运动台结果不能替代 Panda 验收。

## 接口与安装注意

- Shadow 腕手保留 **24 个关节自由度、20 个执行器**。其中 `rh_FFJ0`、`rh_MFJ0`、`rh_RFJ0`、`rh_LFJ0` 是固定腱名称，分别控制对应 `J2 + J1` 的总角；它们不是额外关节，也不强制两个关节角相等。不得改成 24 路独立驱动后仍称为原驱动语义。
- Panda 增加 7 个关节和 7 个执行器；无额外活动法兰时，组合预期为 31 个机器人关节、27 个机器人执行器。物体的自由关节应另计。所有控制和初始化按名称解析，不依赖拼接后数组顺序。
- 无夹爪 Panda 的 `attachment` 位于 `link7` 局部 `pos="0 0 0.107"`，`quat="0.3826834 0 0 0.9238795"`，使用 MuJoCo 的 **wxyz** 顺序。它与完整 `panda.xml` 原夹爪 `hand` 的 `quat="0.9238795 0 0 -0.3826834"` 相差 180° 绕 Z 转动，不能互换。
- 原 Shadow `rh_forearm` 有展示朝向 `quat="0 1 0 1"`；`rh_wrist` 相对原前臂为 `pos="0.01 0 0.21301"`、`quat="1 0 0 1"`。删除前臂后必须重新声明法兰到腕部的安装变换，检查掌面法向、指尖方向及碰撞余量，不能直接沿用原世界展示姿态。
- Panda 原 `home` 的关节目标为 `[0, 0, 0, -1.57079, 0, 1.57079, -0.7853]`。关节 4 的零位不在其限位内；应按 `joint1` 至 `joint7` 名称初始化。原七维 `home` 关键帧及 Shadow 手型关键帧都不能直接作为组合模型的完整状态。

## 验证门槛

先验证依赖完整、组合编译、命名与维数、所有初态限位、安装方向和初始碰撞；再验证有重力时的静态持位、关节跟踪、实际驱动力与饱和。上述门槛通过后，才开展物体接触、抓取和抬升；任务期物体只能由动力学与接触运动。

每一步应保留实际测量及失败记录。目前的仿真动态验证和锤子压按钮任务结果见[验证记录](validation.md)，其结论限于所列场景、参数和试次；来源审查与 XML 统计不能替代这些任务证据，也不能据此宣称真机验证通过。
