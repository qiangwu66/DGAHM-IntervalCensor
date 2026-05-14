
import tensorflow as tf

def g_IC(train_data, Z_validation, Z_test, Lambda_U, Lambda_V, n_node, n_lr, n_epoch, l1_lambda):
    # 加载训练数据和测试数据
    Z_train = tf.convert_to_tensor(train_data['X'], dtype=tf.float32)
    U_train = tf.convert_to_tensor(train_data['U'], dtype=tf.float32)
    V_train = tf.convert_to_tensor(train_data['V'], dtype=tf.float32)
    De1_train = tf.convert_to_tensor(train_data['De1'], dtype=tf.float32)
    De2_train = tf.convert_to_tensor(train_data['De2'], dtype=tf.float32)
    De3_train = tf.convert_to_tensor(train_data['De3'], dtype=tf.float32)
    # g_train_true = tf.convert_to_tensor(train_data['g_Z'], dtype=tf.float32)
    Z_validation = tf.convert_to_tensor(Z_validation, dtype=tf.float32)
    Z_test = tf.convert_to_tensor(Z_test, dtype=tf.float32)
    Lambda_U = tf.convert_to_tensor(Lambda_U, dtype=tf.float32)
    Lambda_V = tf.convert_to_tensor(Lambda_V, dtype=tf.float32)

    # 定义神经网络模型
    class NeuralNetwork(tf.keras.Model):
        def __init__(self, n_node):
            super(NeuralNetwork, self).__init__()
            self.hidden_layers = [
                tf.keras.layers.Dense(n_node, activation='selu'),
                tf.keras.layers.Dense(n_node, activation='selu')
            ]
            self.output_layer = tf.keras.layers.Dense(1, activation='linear')
    
        def call(self, inputs):
            x = inputs
            for layer in self.hidden_layers:
                x = layer(x)
            output = self.output_layer(x)
            return output

    # 自定义损失函数
    def my_loss(De1, De2, De3, Lambda_U, Lambda_V, g_Z):
        Ezg = tf.exp(g_Z)
        loss_fun = -tf.reduce_mean(
            De1 * tf.math.log(1 - tf.exp(-Lambda_U * Ezg) + 1e-5) +
            De2 * tf.math.log(tf.exp(-Lambda_U * Ezg) - tf.exp(-Lambda_V * Ezg) + 1e-5) -
            De3 * Lambda_V * Ezg
        )
        return loss_fun
    # 创建神经网络模型实例
    model = NeuralNetwork(n_node)
    
    # 定义优化器和损失函数
    optimizer = tf.keras.optimizers.Adam(n_lr)
    loss_fn = lambda: my_loss(De1_train, De2_train, De3_train, U_train, V_train, model(Z_train))
    
    # 定义训练步骤
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = loss_fn()
            # 添加 L1 惩罚项
            l1_penalty = tf.reduce_sum([tf.reduce_sum(tf.abs(var)) for var in model.trainable_variables])
            loss += l1_lambda * l1_penalty
    
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 训练模型
    for epoch in range(n_epoch):
        train_step()
    
    # 在训练集上进行预测
    g_train = model(Z_train)
    # 在验证集上进行预测
    g_validation = model(Z_validation)
    # 在测试集上进行预测
    g_test = model(Z_test)
    # 转化为numpy
    g_train = g_train[:, 0].numpy()
    g_validation = g_validation[:, 0].numpy()
    g_test = g_test[:, 0].numpy()

    return {
        'g_train': g_train,
        'g_validation': g_validation,
        'g_test': g_test
    }