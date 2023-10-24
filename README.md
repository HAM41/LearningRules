# Inferring learning rules during decision making 

Project on inferring learning rules underlying sensory decision making in two-alternate force choice tasks. 

We consider decision models making decision $y_t$ from stimulus $x_t$ at trial $t$. So far we have implemented the following behavioral models:
1. Q-learning: 
   - Perception: from noisy percept $m \sim \mathcal{N}(x, \sigma^2)$ form belief $p_R(m_t) = p(s > 0 \mid m_t)$
   - Choice:  Softmax choice between the two $Q$-values. The $Q$-value of choice $c$ is calculated as $Q_c = p_c(m_t)V_c$, from stored value $V_c$ and current belief $p_c(m_t)$.
   - Learning: belief-weighted Q-learning $V_c \gets V_c + \alpha *(r_t - Q_c)$
2. Policy gradient with GLMs:
   - Perception: $x_t$
   - Choice: Bernoulli-GLM with logit link
   - Learning: Policy gradient updates 
   $$
    \Delta w_t = \alpha R(x_t) p(y=R\mid w, x)p(y=L\mid w, x) x
   $$

We perform model fitting with SMC. We plan to deploy the models on learning data from the IBL. 