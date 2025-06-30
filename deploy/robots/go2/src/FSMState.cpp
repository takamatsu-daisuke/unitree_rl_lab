#include "FSMState.h"

std::unique_ptr<unitree::robot::go2::publisher::LowCmd> FSMState::lowcmd = nullptr;
std::shared_ptr<unitree::robot::go2::subscription::LowState> FSMState::lowstate = nullptr;
int FSMState::nq = 12;
