from utils import *
import multiprocessing
if __name__ == '__main__':
    #path  = "/home/yassine/Downloads/facebook"
    G = load_graph()
    p = multiprocessing.Process(target = visualize_graph , args=(G,))
    p.start()
    print("the Main process keep going here ... ! \n")
    print("start the graph embedding here")
    model = graph_embedding(G)
    positive_samples, all_non_edges , negative_samples = generate_positive_edges(G)
    X,y = X_and_y_data_related_to_Our_graph(model,positive_samples, negative_samples)
    print(len(X), len(y))
    print("the X that we will uses is here\n")
    print(X)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
    print("the y that we will uses is here\n")
    print(y)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
    X_train , X_test ,y_train ,  y_test = spliter_of_data(X,y)

    X_train_scaled , X_test_scaled = scaler(X_train, X_test)
    best_clf_name , best_score  = classifier_models(classifiers, X_train_scaled , X_test_scaled ,y_train ,  y_test)
    

    
    




    

