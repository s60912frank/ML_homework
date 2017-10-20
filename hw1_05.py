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
        <div style="height:900px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:900px;height:920px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
###### Do not modify  here ######

###### Implement Data Preprocess here ######
housing = fetch_california_housing()
print("Shape of dataset:", housing.data.shape)
print("Shape of label:", housing.target.shape)

###### Constants ######
LEARNING_RATE = 0.00000000003
#LEARNING_RATE = 0.0001
BATCH_SIZE = 48
DATA_LENGTH = len(housing.data)
print(DATA_LENGTH)


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
            counter += 1
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
    batch_collection.append(one_batch)
    output_collection.append(one_output)
    
    return batch_collection, output_collection



###### Implement Data Preprocess here ######
X = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 8), name="X")
Y = tf.placeholder(tf.float32, name="Y")
#w = tf.Variable(np.random.rand(8, 1).astype('f'), name="weights")
w = tf.Variable(np.zeros((8, 1)).astype('f'), name="weights")
b = tf.Variable(0.0, name="bias")

Y_hat = tf.add(tf.matmul(X, w), b)
loss = tf.reduce_sum(tf.square(Y - Y_hat)) / BATCH_SIZE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

## Shuffle Data
print(housing.data.shape[0])
rnd_order = np.arange(housing.data.shape[0])
np.random.shuffle(rnd_order)
print(rnd_order)
housing.data = housing.data[rnd_order]
housing.target = housing.target[rnd_order]


###### Start TF session ######
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_bat, y_bat = make_batch(housing.data, housing.target)
    print(len(x_bat))
    for i in range(300):
        cost = 0
        for j in range(int(len(x_bat) * 0.9)):
        #for j in range(int(len(x_bat) * 0.9), len(x_bat)):
            x = x_bat[j]
            y = y_bat[j]
            sess.run(optimizer, feed_dict={X: x, Y: y}) #actual training
            cost += sess.run(loss, feed_dict={X: x, Y: y}) #
        print("========================\nepoch: {0}, \nCost: {1}, \nWeight: {2}, \nBias: {3}".format(i, cost / len(x_bat), sess.run(w).reshape((1, 8)), sess.run(b)))
        
        #test
        err_rate = 0.0
        for k in range(int(len(x_bat) * 0.9), len(x_bat)):
            x = x_bat[k]
            y = y_bat[k]
            y_prid = sess.run(Y_hat, feed_dict={X: x})
            err_rate += np.sum(np.abs(y - y_prid) / y) / BATCH_SIZE #treudiv
        print(err_rate / len(x_bat) * 0.1)
    tf.summary.FileWriter("logs/", tf.get_default_graph().as_graph_def())
    show_graph(tf.get_default_graph().as_graph_def())
###### Start TF session ######
