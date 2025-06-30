#pragma once

#include "param.h"
#include "FSM/BaseState.h"
#include "unitree/dds_wrapper/robots/go2/go2.h" // The idl of H1 is same as Go2.

using namespace unitree::robot;

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->joystick.LT.pressed && lowstate->joystick.B.on_pressed; },
                1 // 1 must be Passive state
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<go2::publisher::LowCmd> lowcmd;
    static std::shared_ptr<go2::subscription::LowState> lowstate;
    static int nq;
};