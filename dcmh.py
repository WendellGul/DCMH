from config import opt
from data_handler import load_data
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from models import ImgModule, TxtModule
from utils import calc_map


def train(**kwargs):
    opt.parse(kwargs)

    images, tags, labels = load_data(opt.data_path)
    y_dim = tags.shape[1]

    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    train_L = torch.from_numpy(L['train'])
    train_x = torch.from_numpy(X['train'])
    train_y = torch.from_numpy(Y['train'])

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    num_train = train_x.shape[0]

    Sim = calc_neighbor(train_L, train_L)

    F = Variable(torch.randn(opt.bit, num_train))
    G = Variable(torch.randn(opt.bit, num_train))

    if opt.use_gpu:
        F = F.cuda()
        G = G.cuda()

    batch_size = opt.batch_size

    lr = opt.lr
    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)

    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch)
    for epoch in range(opt.max_epoch):
        lr = learning_rate[epoch]

        # train image net
        for i in range(num_train // batch_size):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]

            sample_L = train_L[ind, :]
            image = Variable(train_x[ind].type(torch.float))
            if opt.use_gpu:
                image = image.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, sample_L)  # S: (batch_size, batch_size)
            cur_f = img_model(image)  # cur_f: (batch_size, bit)
            F[:, ind] = cur_f
            B = torch.sign(F + G)

            # calculate loss
            # theta_x: (batch_size, batch_size)
            theta_x = 1.0 / 2 * torch.matmul(cur_f, G[ind, :].transpose(0, 1))
            logloss_x = -torch.sum(torch.mul(S, theta_x) - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(torch.sum(F, dim=0), 2))
            loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x
            loss_x /= batch_size

            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()

        # train txt net
        for i in range(num_train // batch_size):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]

            sample_L = train_L[ind, :]
            text = Variable(train_y[ind].type(torch.float))
            if opt.use_gpu:
                image = image.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, sample_L)  # S: (batch_size, batch_size)
            cur_g = txt_model(text)  # cur_f: (batch_size, bit)
            G[:, ind] = cur_g
            B = torch.sign(F + G)

            # calculate loss
            # theta_y: (batch_size, batch_size)
            theta_y = 1.0 / 2 * torch.matmul(F[ind, :], cur_g.transpose(0, 1))
            logloss_y = -torch.sum(torch.mul(S, theta_y) - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(torch.sum(G, dim=0), 2))
            loss_y = logloss_y + opt.gamma * quantization_y + opt.eta * balance_y
            loss_y /= (num_train * batch_size)

            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()

        img_model.save(img_model.module_name + '_' + str(epoch) + '.pth')
        txt_model.save(txt_model.module_name + '_' + str(epoch) + '.pth')

        loss = calc_loss(B, F, G, Sim, opt.gamma, opt.eta)

        print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss, lr))

    print('...training procedure finish')
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map(qBX, rBX, query_L, retrieval_L)
    mapt2i = calc_map(qBY, rBY, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))


def split_data(images, tags, labels):
    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.int)
    return Sim


def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F.transpose(0, 1), G) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.power(B - F, 2) + torch.power(B - G, 2))
    term3 = torch.sum(torch.power(F.sum(dim=0), 2) + torch.power(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float32)
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float32)

        cur_f = img_model(image)
        B[ind, :] = cur_f
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float32)
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].type(torch.float32)

        cur_g = txt_model(text)
        B[ind, :] = cur_g
    B = torch.sign(B)
    return B


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
        usage : python file.py <function> [--args=value]
        <function> := train | test | help
        example: 
                python {0} train --lr=0.01
                python {0} help
        avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    train()
