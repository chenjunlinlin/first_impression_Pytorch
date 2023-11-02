from torchview import draw_graph

def get_model_graph(save_path, filename, model, shape):
    """
    Get the model overview
    """
    model_graph = draw_graph(model, input_size=shape, expand_nested=True, save_graph=True, filename=filename, directory=save_path)

    model_graph.visual_graph
