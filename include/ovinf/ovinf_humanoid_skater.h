/**
 * @file ovinf_humanoid_skater.h
 * @brief ovinf for T1 humanoid skateboarding policy (HUSKY port).
 *
 * 配套模型：humanoid_skateboarding 训练导出的 policy_timemajor_*.onnx
 * （图内已含：时间主序历史重排 + 观测归一化，输入 [1, 395] = 79维单帧 × 5帧，
 *  时间主序、旧→新，直接喂 HistoryBuffer::GetObsHistory() 即可）。
 *
 * 单帧 79 维布局（按训练 obs 组拼接顺序）：
 *   [0:2]   command      = (cmd_vx × 2.0, cmd_heading × 1.0)
 *   [2]     heading      = yaw × 1/π
 *   [3:6]   ang_vel      = IMU 角速度(本体系) × 0.25
 *   [6:9]   proj_gravity
 *   [9:32]  joint_pos    = (关节角 − 默认位姿) × 1.0
 *   [32:55] joint_vel    = 关节角速度 × 0.05
 *   [55:78] last_action  = 上一步网络原始输出
 *   [78]    phase        = 步态相位 0~1（内部时钟，周期 6s）
 *
 * 关节顺序与默认位姿由 YAML policy_joint_names / policy_default_position 给出，
 * 应与训练侧 T1_JOINT_NAMES（= bitbot 部署栈顺序，Elbow_Yaw=弯曲/Elbow_Pitch=自转）一致。
 */

#ifndef OVINF_HUMANOID_SKATER_H
#define OVINF_HUMANOID_SKATER_H

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <openvino/openvino.hpp>
#include <optional>
#include <string>
#include <thread>

#include "atomicops.h"
#include "ovinf.hpp"
#include "readerwriterqueue.h"
#include "utils/csv_logger.hpp"
#include "utils/history_buffer.hpp"
#include "utils/realtime_setting.h"

namespace ovinf {

class HumanoidSkaterPolicy : public BasePolicy<float> {
  using MatrixT = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<float, Eigen::Dynamic, 1>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 public:
  HumanoidSkaterPolicy() = delete;
  ~HumanoidSkaterPolicy();

  HumanoidSkaterPolicy(const YAML::Node &config);

  /**
   * @brief Policy warmup
   *
   * @param[in] obs_pack Robot observation
   * @return Is warmup done successfully.
   */
  virtual bool WarmUp(RobotObservation<float> const &obs_pack) final;

  /**
   * @brief Set observation, run inference.
   *
   * @param[in] obs_pack Robot observation
   * @return Is inference started immidiately.
   */
  virtual bool InferUnsync(RobotObservation<float> const &obs_pack) final;

  /**
   * @brief Get resulting target_joint_pos
   *
   * @param[in] timeout Timeout in microseconds
   */
  virtual std::optional<VectorT> GetResult(const size_t timeout = 100) final;

  virtual void PrintInfo() final;

 private:
  void WorkerThread();
  void CreateLog(YAML::Node const &config);
  void WriteLog(RobotObservation<float> const &obs_pack);
  VectorT BuildSingleObs(RobotObservation<float> const &obs_pack);

 private:
  // Threading
  std::atomic<bool> inference_done_{false};
  std::atomic<bool> exiting_{false};
  std::thread worker_thread_;

  // OpenVINO inference
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
  ov::Output<const ov::Node> input_info_;

  // Infer data
  std::map<std::string, size_t> joint_names_;
  float cycle_time_;
  VectorT joint_default_position_;

  size_t single_obs_size_;
  size_t obs_buffer_size_;
  size_t action_size_;

  VectorT action_scale_;  // 逐关节动作缩放（YAML policy_action_scale，缺省用标量 action_scale）
  float action_scale_scalar_;
  float obs_scale_ang_vel_;
  float obs_scale_dof_pos_;
  float obs_scale_dof_vel_;
  float obs_scale_proj_gravity_;
  float obs_scale_heading_;
  float obs_scale_command_x_;
  float obs_scale_command_heading_;
  float clip_action_;

  // Buffer（时间主序历史，喂给图内已重排的 ONNX）
  moodycamel::ReaderWriterQueue<VectorT> input_queue_;
  std::shared_ptr<HistoryBuffer<float>> obs_buffer_;
  VectorT last_action_;
  VectorT latest_target_;

  // Gait phase clock（训练中始终推进：周期 6s，phase = t/cycle mod 1）
  double current_phase_time_ = 0.0;
  bool use_absolute_clock_ = true;
  std::chrono::steady_clock::time_point phase_start_time_;
  float control_period_ = 0.02f;

  std::chrono::high_resolution_clock::time_point infer_start_time_;
  std::chrono::high_resolution_clock::time_point infer_end_time_;

  // Logger
  bool log_flag_ = false;
  CsvLogger::Ptr csv_logger_;
  std::string log_name_;
  float inference_time_ = 0;

  // Realtime
  size_t stick_to_core_ = 0;
};
}  // namespace ovinf

#endif  // !OVINF_HUMANOID_SKATER_H
