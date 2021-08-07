import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import dgl

def train_cora_node_classification(model, G):
    n_epochs = 100

    optimizer = optim.Adam(model.parameters(), lr=.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_test_acc = 0, 0


    features = G.ndata['feat']
    labels = G.ndata['label']
    train_mask = G.ndata['train_mask']
    val_mask = G.ndata['val_mask']
    test_mask = G.ndata['test_mask']

    for epoch in range(n_epochs):
        # forward
        logits = model(G, features)

        # loss
        loss = criterion(logits[train_mask], labels[train_mask])

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # eval
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            train_acc = (predictions[train_mask] ==
                        labels[train_mask]).float().mean()
            val_acc = (predictions[val_mask] == labels[val_mask]).float().mean()
            test_acc = (predictions[test_mask] == labels[test_mask]).float().mean()

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

        if not epoch % 5:
            print(f'In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})')


def create_heterograph(n_users=1000, n_items=500,
                       n_follows=3000, n_clicks=5000,
                       n_dislikes=500, n_hetero_features=10,
                       n_user_classes=5, n_max_clicks=10):

    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)

    hetero_graph = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src),
        ('user', 'click', 'item'): (click_src, click_dst),
        ('item', 'clicked-by', 'user'): (click_dst, click_src),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
        ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})

    hetero_graph.n_users = n_users
    hetero_graph.n_items = n_items
    hetero_graph.n_follows = n_follows
    hetero_graph.n_clicks = n_clicks
    hetero_graph.n_dislikes = n_dislikes
    hetero_graph.n_hetero_features = n_hetero_features
    hetero_graph.n_user_classes = n_user_classes
    hetero_graph.n_max_clicks = n_max_clicks

    hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
    hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
    hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
    hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()
    # randomly generate training masks on user nodes and click edges
    hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
    hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)

    return hetero_graph


def create_edge_pred_graph(n_nodes, n_edges,
                           n_node_features, n_edge_features):
    src = np.random.randint(0, n_nodes, n_edges // 2)
    dst = np.random.randint(0, n_nodes, n_edges // 2)
    # сделаем граф "неориентированным"
    edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
    # добавим фичи на узлы и ребра
    edge_pred_graph.ndata['feature'] = torch.randn(n_nodes, n_node_features)
    edge_pred_graph.edata['feature'] = torch.randn(n_edges, n_edge_features)
    # значения для прогнозирования на ребра
    edge_pred_graph.edata['label'] = torch.randn(n_edges)
    edge_pred_graph.edata['label_class'] = torch.zeros(n_edges).bernoulli(.5)
    # маска для обучающей выборки
    edge_pred_graph.edata['train_mask'] = torch.zeros(n_edges, dtype=torch.bool).bernoulli(0.6)
    return edge_pred_graph

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes