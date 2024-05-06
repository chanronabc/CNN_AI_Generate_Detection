import matplotlib.pyplot as plt
import networkx as nx

def draw_cnn_horizontal():
    # Create a directed graph
    G = nx.DiGraph()

    # Nodes represent layers in the CNN
    layers = [
        "Input (3C)",
        "Conv1 \n(16C, 3x3, P=1)\n + ReLU",
        "MaxPool1 \n(2x2, S=2)",
        "Conv2 \n(32C, 3x3, P=1)\n + ReLU",
        "MaxPool2 \n(2x2, S=2)",
        "Conv3 \n(64C, 3x3, P=1)\n + ReLU",
        "MaxPool3 \n(2x2, S=2)",
        "Flatten",
        "FC1 \n(2048 -> 128)\n + ReLU",
        "Dropout \n(0.5)",
        "FC2 \n(128 -> 2)"
    ]

    # Add nodes with the layer descriptions
    for i, layer in enumerate(layers):
        G.add_node(layer, pos=(i, 0))

    # Connect nodes
    for i in range(len(layers) - 1):
        G.add_edge(layers[i], layers[i + 1])

    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=5000, node_color='skyblue', font_size=8.5, font_weight='bold', horizontalalignment='center')
    plt.title('Horizontal CNN Architecture with NetworkX')
    plt.show()

# Execute the function to draw the CNN architecture horizontally
draw_cnn_horizontal()
