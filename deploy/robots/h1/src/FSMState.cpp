#include "FSMState.h"

std::unique_ptr<unitree::robot::go2::publisher::LowCmd> FSMState::lowcmd = nullptr;
std::shared_ptr<unitree::robot::go2::subscription::LowState> FSMState::lowstate = nullptr;
int FSMState::nq = 19 + 1; // max index of joint;  motor_cmd[9] not used