// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/action_manager.h"

namespace isaaclab
{

class JointAction : public ActionTerm
{
public:
    JointAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :ActionTerm(cfg, env)
    {
        _num_joints = env->robot->data.joint_ids_map.size();
        _raw_actions.resize(_num_joints, 0.0f);
        _processed_actions.resize(_num_joints, 0.0f);
        _scale = cfg["scale"].as<std::vector<float>>();
        _offset = cfg["offset"].as<std::vector<float>>();

        if(!cfg["clip"].IsNull())
        {
            _clip = cfg["clip"].as<std::vector<std::vector<float> >>();
        }
    }

    void process_actions(std::vector<float> actions)
    {
        _raw_actions = actions;
        for(int i(0); i<_num_joints; ++i)
        {
            _processed_actions[i] = _raw_actions[i] * _scale[i] + _offset[i];
        }
        if(!_clip.empty())
        {
            for(int i(0); i<_num_joints; ++i)
            {
                _processed_actions[i] = std::clamp(_processed_actions[i], _clip[i][0], _clip[i][1]);
            }
        }
    }

    int action_dim() 
    {
        return _num_joints;
    }

    std::vector<float> raw_actions() 
    {
        return _raw_actions;
    }
    
    std::vector<float> processed_actions() 
    {
        return _processed_actions;
    }

    void reset()
    {
        _raw_actions.assign(_num_joints, 0.0f);
    }

protected:
    int _num_joints;

    std::vector<float> _raw_actions;
    std::vector<float> _processed_actions;

    std::vector<float> _scale;
    std::vector<float> _offset;
    std::vector<std::vector<float> > _clip;
};


class JointPositionAction : public JointAction
{
public:
    JointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    :JointAction(cfg, env)
    {
    }
};

REGISTER_ACTION(JointPositionAction);

};