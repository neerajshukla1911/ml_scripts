import tensorflow as tf
import keras

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # set Keras learning phase to train
            keras.backend.set_learning_phase(1)
            # do not initialize variables on the fly
            keras.backend.manual_variable_initialization(True)

            # Build Keras model
            model = ...

            # keras model predictions
            preds = model.output
            # placeholder for training targets
            targets = tf.placeholder(...)
            # our categorical crossentropy loss
            xent_loss = tf.reduce_mean(
                keras.objectives.categorical_crossentropy(targets, preds))

            # we create a global_step tensor for distributed training
            # (a counter of iterations)
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # apply regularizers if any
            if model.regularizers:
                total_loss = xent_loss * 1.  # copy tensor
                for regularizer in model.regularizers:
                    total_loss = regularizer(total_loss)
            else:
                total_loss = xent_loss

            # set up TF optimizer
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate, decay=0.9, momentum=FLAGS.momentum, epsilon=1e-8)

            # Set up model update ops (batch norm ops).
            # The gradients should only be computed after updating the moving average
            # of the batch normalization parameters, in order to prevent a data race
            # between the parameter updates and moving average computations.
            with tf.control_dependencies(model.updates):
                barrier = tf.no_op(name='update_barrier')

            # define gradient updates
            with tf.control_dependencies([barrier]):
                grads = optimizer.compute_gradients(
                    total_loss,
                    model.trainable_weights,
                    gate_gradients=tf.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False)

            # define train tensor
            train_tensor = tf.with_dependencies([grad_updates],
                                                total_loss,
                                                name='train')

            # blah blah
            saver = tf.train.Saver()
            summary_op = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

            # Create a "supervisor", which oversees the training process.
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                     logdir="/tmp/train_logs",
                                     init_op=init_op,
                                     summary_op=summary_op,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=600)

            # The supervisor takes care of session initialization, restoring from
            # a checkpoint, and closing when done or an error occurs.
            with sv.managed_session(server.target) as sess:
                # Loop until the supervisor shuts down or 1000000 steps have completed.
                step = 0
                while not sv.should_stop() and step < 1000000:
                    # Run a training step asynchronously.
                    # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                    # perform *synchronous* training.

                    # feed_dict must contain the model inputs (the tensors listed in model.inputs)
                    # and the "targets" placeholder we created ealier
                    # it's a dictionary mapping tensors to batches of Numpy data
                    # like: feed_dict={model.inputs[0]: np_train_data_batch, targets: np_train_labels_batch}
                    loss_value, step_value = sess.run([train_op, global_step], feed_dict={...})

            # Ask for all the services to stop.
            sv.stop()


if __name__ == "__main__":
    tf.app.run()