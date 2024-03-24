import numpy as np
import torch
import torch.nn as nn
import neurogym as ngym

# https://neurogym.github.io/envs/index.html

def get_task_list():
    # 可用的任务
    regist_tasks ={
        'AntiReach-v0': 'Anti-response task.',
        'ContextDecisionMaking-v0': 'Context-dependent decision-making task.',
        'DelayedComparison-v0': 'Delayed comparison task.',
        'DelayMatchCategory-v0': 'Delayed match-to-category task.',
        'DelayMatchSample-v0': 'Delayed match-to-sample task.',
        'DelayMatchSampleDistractor1D-v0': 'Delayed match-to-sample with multiple, potentially repeating distractors.',
        'DelayPairedAssociation-v0': 'Delayed paired-association task.',
        'DualDelayMatchSample-v0': 'Two-item Delay-match-to-sample.',
        'GoNogo-v0': 'Go/No-go task.',
        'HierarchicalReasoning-v0': 'Hierarchical reasoning of rules.',
        'IntervalDiscrimination-v0': 'Comparing the time length of two stimuli.',
        'MotorTiming-v0': 'Agents have to produce different time intervals using different effectors (actions).',
        'MultiSensoryIntegration-v0': 'Multi-sensory integration.',
        'OneTwoThreeGo-v0': 'Agents reproduce time intervals based on two samples.',
        'PerceptualDecisionMaking-v0': 'Two-alternative forced choice task in which the subject has to integrate two stimuli to decide which one is higher on average.',
        'PerceptualDecisionMakingDelayResponse-v0': 'Perceptual decision-making with delayed responses.',
        'PostDecisionWager-v0': 'Post-decision wagering task assessing confidence.',
        'ProbabilisticReasoning-v0': 'Probabilistic reasoning.',
        'PulseDecisionMaking-v0': 'Pulse-based decision making task.',
        'Reaching1D-v0': 'Reaching to the stimulus.',
        'Reaching1DWithSelfDistraction-v0': 'Reaching with self distraction.',
        'ReadySetGo-v0': 'Agents have to measure and produce different time intervals.',
        'SingleContextDecisionMaking-v0': 'Context-dependent decision-making task.',
    }

    # 未实现的任务
    Not_implemented_tasks = {
        'Bandit-v0': 'Multi-arm bandit task.',
        'DawTwoStep-v0': 'Daw Two-step task.',
        'EconomicDecisionMaking-v0': 'Economic decision making task.',
        'ReachingDelayResponse-v0': 'Reaching task with a delay period.',
        'psychopy.RandomDotMotion-v0': 'Two-alternative forced choice task in which the subject has to integrate '
                                       'two stimuli to decide which one is higher on average.',
        'psychopy.VisualSearch-v0': 'Visual search task.',
    }

    return regist_tasks.keys(), Not_implemented_tasks.keys()



def create_dataset(task, seq_len, batch_size, kwargs):
    '''
    :param task: 任务名称
    :param seq_len: 任务序列长度
    :param batch_size: 批次大小
    :param kwargs: 任务参数，多少长度dt合并为一个时间步
    :return: dataset, 数据集 data loader
            ob_size, 观测空间大小, 网络输入大小
            ac_size, 动作空间大小, 网络输出大小
    '''
    # 判断task是否在注册任务中
    regist_tasks, _ = get_task_list()
    assert task in regist_tasks, 'task {} is not in regist_tasks'.format(task)
    dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size,
                       seq_len=seq_len)
    env = dataset.env
    ob_size = env.observation_space.shape[0]
    ac_size = env.action_space.n
    print('observation_size (netwrok input dim): {}, action_size (network output dim): {}'.format(ob_size, ac_size))
    return dataset, ob_size, ac_size

if __name__ == '__main__':
    task = 'ContextDecisionMaking-v0'
    seq_len = 100
    batch_size = 32
    kwargs = {'dt': 100}
    dataset, ob_size, ac_size = create_dataset(task, seq_len, batch_size, kwargs)
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float)
    labels = torch.from_numpy(labels).type(torch.long)
    print(inputs.shape, labels.shape)
    print(labels[0:5, :])

