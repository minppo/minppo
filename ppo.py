import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

pi_lr = 3e-4
v_lr = 1e-3
kl_target = 0.01
clip = 0.2
sgd_pi_steps = 80
sgd_v_steps = 80
gae_lam = 0.97
discount = 0.99

num_epochs = 100
exp_per_epoch = 8
max_exp_steps = 200
steps_per_epoch = 4000

envname = 'CartPole-v0'
    
def ppo():
    env = gym.make(envname)

    ob_shape = env.observation_space.shape
    ac_shape = env.action_space.shape
    ob_ph = tf.placeholder(tf.float32, shape=(None, *ob_shape), name='ob')
    ac_dtype = tf.float32 if isinstance(env.action_space, Box) else tf.int32
    ac_ph = tf.placeholder(ac_dtype, shape=(None, *ac_shape), name='ac')

    old_logprob_ph = tf.placeholder(tf.float32, shape=(None,), name='old_logprob')
    adv_ph = tf.placeholder(tf.float32, shape=(None,), name='adv')
    ret_ph = tf.placeholder(tf.float32, shape=(None,), name='ret')


    sampled_ac, logprob, sampled_logprob, ob_val = mlp_actor_critic(ob_ph, ac_ph,
                                                    action_space=env.action_space)

    value_loss = tf.reduce_mean((ret_ph - ob_val)**2)
    train_critic = tf.train.AdamOptimizer(v_lr).minimize(value_loss)

    pi_ratio = tf.exp(logprob - old_logprob_ph)
    pi_loss = (-1) * adv_ph * tf.where(adv_ph > 0, 
                            tf.minimum(pi_ratio, 1+clip), 
                            tf.maximum(pi_ratio, 1-clip))
    train_actor = tf.train.AdamOptimizer(pi_lr).minimize(pi_loss)

    kl_approx = tf.reduce_mean(old_logprob_ph - logprob)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(num_epochs):
        ### SAMPLE TRAJECTORIES
        buf = Buffer(ob_shape, ac_shape)
        epo_re = 0

        ob = env.reset()
        done = False
        num_exps = 0
        for step in range(steps_per_epoch):
            ac, v, logp = sess.run([sampled_ac, ob_val, sampled_logprob], 
                                   feed_dict={ob_ph : [ob]})
            new_ob, re, done, info = env.step(ac[0])
            buf.append((ob, ac, re, v, logp))
            ob = new_ob
            epo_re += re
            if done or step == max_exp_steps-1:
                buf.close_trajectory()
                ob = env.reset()
                done = False
                num_exps += 1

        print(f'Avg reward: {epo_re / num_exps:.2f}')
        ### UPDATE WEIGHTS
        obs, acs, advs, rtgs, logps = buf.get()
        feed_dict = {ob_ph:obs, ac_ph:acs, adv_ph:advs, ret_ph:rtgs, old_logprob_ph:logps}
        t_kl = 0
        for _ in range(sgd_pi_steps):
            _, kl = sess.run([train_actor, kl_approx], feed_dict=feed_dict)
            t_kl += kl
            if kl > 1.5 * kl_target:
                break
        for _ in range(sgd_v_steps):
            sess.run(train_critic, feed_dict=feed_dict)

    sess.close()
    env.close()

def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None,  action_space=None):

    if isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


class Buffer:
    def __init__(self, ob_shape, ac_shape):
        max_size = steps_per_epoch
        self.ob   = np.zeros(shape=(max_size, *ob_shape), dtype=np.float32)
        self.ac   = np.zeros(shape=(max_size, *ac_shape), dtype=np.float32)
        self.v    = np.zeros(shape=(max_size,), dtype=np.float32)
        self.re   = np.zeros(shape=(max_size,), dtype=np.float32)
        self.rtg  = np.zeros(shape=(max_size,), dtype=np.float32)
        self.adv  = np.zeros(shape=(max_size,), dtype=np.float32)
        self.logp = np.zeros(shape=(max_size,), dtype=np.float32)

        self.pos = 0
        self.path_start = self.pos

    def append(self, exp):
        ob, ac, re, v, logp = exp
        self.ob[self.pos] = ob
        self.ac[self.pos] = ac
        self.re[self.pos] = re
        self.v[self.pos] = v
        self.logp[self.pos] = logp
        self.pos += 1

    def close_trajectory(self):
        rev_rtgs = []
        rev_advs = []
        rtg = 0
        adv = 0

        path = slice(self.path_start, self.pos)

        res = self.re[path]
        vs = np.append(self.v[path], 0)
        for i in reversed(range(len(res))):
            rtg = res[i] + discount*rtg
            rev_rtgs.append(rtg)

            dv = (res[i] + discount*vs[i+1]) - vs[i]
            adv = dv + gae_lam*discount*adv
            rev_advs.append(adv)

        rtgs = list(reversed(rev_rtgs))
        advs = list(reversed(rev_advs))
        self.rtg[path] = rtgs
        self.adv[path] = advs

        self.path_start = self.pos

    def get(self):
        self.adv = (self.adv - np.mean(self.adv)) / np.std(self.adv)
        return [self.ob, self.ac, self.adv, self.rtg, self.logp]


if __name__ == '__main__':
    ppo()
