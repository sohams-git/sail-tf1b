import numpy as np
import tensorflow as tf
from stable_baselines.td3.td3 import TD3 as TD3_Vanilla

class TD3Pref(TD3_Vanilla):
    def __init__(self, *args, ref_lambda=0.0, pref_rm=None,
                 pref_expect_obs_dim=17, pref_norm="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_lambda = float(ref_lambda)
        self.pref_rm = pref_rm
        self.pref_expect_obs_dim = pref_expect_obs_dim
        self.pref_norm = pref_norm

    def _setup_model(self):
        super()._setup_model()
        with tf.variable_scope("pref_aux", reuse=tf.AUTO_REUSE):
            self.y_pref_ph = tf.placeholder(tf.float32, shape=(None, 1), name="y_pref")
            self.lambda_ref_ph = tf.placeholder_with_default(self.ref_lambda, shape=(), name="lambda_ref")
            q1 = getattr(self, "qf1", None) or getattr(self, "qf1_output")
            q2 = getattr(self, "qf2", None) or getattr(self, "qf2_output")
            ref_l2 = 0.5 * tf.reduce_mean(tf.square(q1 - self.y_pref_ph)) + \
                     0.5 * tf.reduce_mean(tf.square(q2 - self.y_pref_ph))
            tf.summary.scalar("loss/ref_l2", ref_l2)
            critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/critic")
            self.pref_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize_critic_pref = self.pref_optimizer.minimize(
                self.lambda_ref_ph * ref_l2, var_list=critic_params)

    @staticmethod
    def _zscore(x, eps=1e-8):
        mu, sd = np.mean(x), np.std(x)
        return (x - mu) / (sd + eps)

    def _train_step(self, obs_t, action, reward, obs_tp1, done, weights, step, writer, local=False):
        out = super(TD3Pref, self)._train_step(obs_t, action, reward, obs_tp1, done, weights, step, writer, local)
        if self.pref_rm is None or self.ref_lambda == 0.0:
            return out
        # BPref model is obs+act; passing next_obs is harmless (loader tries obs->(obs,act)->(obs,act,next_obs))
        r_pref = self.pref_rm.reward(obs_t, action, obs_tp1).reshape(-1, 1)
        r_pref = self._zscore(r_pref)
        feed = {
            self.obs_ph: obs_t,
            self.actions_ph: action,
            self.y_pref_ph: r_pref,
            self.lambda_ref_ph: self.ref_lambda
        }
        self.sess.run(self.optimize_critic_pref, feed_dict=feed)
        return out
