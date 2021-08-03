import torch
import torch.nn as nn
import torch.optim as optim


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
