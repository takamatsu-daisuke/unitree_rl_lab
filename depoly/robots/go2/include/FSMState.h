#pragma once

#include "param.h"
#include "FSM/BaseState.h"
#include "unitree/dds_wrapper/robots/go2/go2.h"

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

    int pre_check()
    {
        auto & joy = lowstate->joystick;
        if(joy.LT.pressed && joy.B.on_pressed) { 
            return FSMStringMap.right.at("Passive");
        }
        return 0;
    }

    static std::unique_ptr<go2::publisher::LowCmd> lowcmd;
    static std::shared_ptr<go2::subscription::LowState> lowstate;
    static int nq;
};