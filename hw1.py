import tensorflow as tf
import numpy as np 
from sklearn.datasets import fetch_california_housing
from IPython.display import clear_output, Image, display, HTML

###### Do not modify here ###### 
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = graph_def
    #strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
###### Do not modify  here ######

###### Implement Data Preprocess here ######
housing = fetch_california_housing()
print("Shape of dataset:", housing.data.shape)
print("Shape of label:", housing.target.shape)

###### Constants ######
LEARNING_RATE = 0.000000003
#BATCH_SIZE = 100
DATA_LENGTH = len(housing.data)
print(DATA_LENGTH)

"""
# make batch here
def make_batch(x, y_hat):
    #global BATCH_SIZE            
    batch_collection = []
    one_batch = np.zeros((BATCH_SIZE, 8))
    output_collection = []
    one_output = []
    counter = 0
    for d in range(DATA_LENGTH):
        if counter < BATCH_SIZE:
            one_batch[counter] = x[d]
            counter++
            one_output.append(y_hat[d])
        else:
            batch_collection.append(one_batch)
            output_collection.append(one_output)
            #重新init
            one_batch = np.zeros((BATCH_SIZE, 8))
            one_batch[0] = x[d]
            counter = 1
            one_output = [y_hat[d]]
    #最後把不足batch的也放進去
    if(len(one_batch) > 0):
        batch_collection.append(one_batch)
        output_collection.append(one_output)
    return batch_collection, output_collection
"""


###### Implement Data Preprocess here ######
X = tf.placeholder(tf.float32, shape=(1, 8), name="X")
Y = tf.placeholder(tf.float32, name="Y")
#w = tf.Variable(np.random.rand(8, 1).astype('f'), name="weights")
w = tf.Variable(np.zeros((8, 1)).astype('f'), name="weights")
b = tf.Variable(0.0, name="bias")

Y_hat = tf.add(tf.matmul(X, w), b)
loss = tf.square(Y - Y_hat, name="loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

###### Start TF session ######
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #x_bat, y_bat = make_batch(housing.data, housing.target)
    for i in range(100):
        cost = 0
        for j in range(DATA_LENGTH): #要改
            x = housing.data[j].reshape((1, 8))
            y = housing.target[j]
            sess.run(optimizer, feed_dict={X: x, Y: y}) #actual training
            cost += sess.run(loss, feed_dict={X: x, Y: y})[0][0]
        print("========================\nepoch: {0}, \nCost: {1}, \nWeight: {2}, \nBias: {3}".format(i, cost / DATA_LENGTH, sess.run(w).reshape((1, 8)), sess.run(b)))
    show_graph(tf.get_default_graph().as_graph_def())
###### Start TF session ######
