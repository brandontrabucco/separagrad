import tensorflow as tf
import numpy as np
import argparse


def main(args):

    initializer = tf.contrib.layers.xavier_initializer()

    graph = tf.Graph()
    with graph.as_default():

        logits = tf.get_variable(name="logits",
            shape=[args.vocab_size], initializer=initializer)
        probabilities = tf.nn.softmax(logits)

        embeddings = tf.get_variable(name="embeddings",
            shape=[args.vocab_size, args.embedding_size], initializer=initializer)
        target_vector = tf.stop_gradient(
            tf.expand_dims(embeddings[0, :] + embeddings[-1, :], 0))

        losses = tf.reduce_sum(tf.squared_difference(
            target_vector, embeddings), 1)
        expected_loss = tf.reduce_sum(losses * probabilities)

        loss_gradient = tf.gradients(losses, embeddings)[0]
        expected_gradient = tf.reduce_sum(
            loss_gradient * tf.expand_dims(probabilities, 1), 0)

        expected_vector = tf.reduce_sum(embeddings * tf.expand_dims(
            probabilities, 1), 0)
        logits_gradient = tf.concat([
            tf.gradients(x, logits) for x in tf.unstack(expected_vector)], 0)

        # One of the probabilities always seems to dominate the learning process
        separated_gradient = tf.tensordot(expected_gradient, logits_gradient, 1)
        update_op = tf.assign(logits, 
            logits - 0.001 * separated_gradient * (1 - probabilities))
        init_op = tf.global_variables_initializer()

    sess = tf.Session(graph=graph)
    sess.run(init_op)
    for i in range(100000):
        if i % 10000 == 0:
            print("Iteration %d probs were %s." % (i, str(sess.run(probabilities))))
        sess.run(update_op)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interpreter")
    parser.add_argument("--vocab_size", type=int, default=5)
    parser.add_argument("--embedding_size", type=int, default=3)

    args = parser.parse_args()
    main(args)