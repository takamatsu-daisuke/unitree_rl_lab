// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"

class State_Passive : public FSMState
{
public:
    State_Passive(int state) : FSMState(state, "Passive") {} 

    void enter()
    {
        // set gain
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        for(int i(0); i < nq; ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.mode() = 1;
            motor.kp() = 0;
            motor.kd() = kd[i];
            motor.tau() = 0;
        }
    }

    void run()
    {
        for(int i(0); i < nq; ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }
    }
};
