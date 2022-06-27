# GL-Vcap
主要是探究vcap的作用，涉及到不同版本的代码

1. GAT-LSTM-F(1):block输出F(1)，并且取最后一个时间片的值： 
2. GAT-LSTM-F(4)-L:block输出F(4)经过L变为F(1)，并且取最后一个时间片的值： 
3. GAT-LSTM-F(1)-Conv：block输出F(1)，==卷积==取一个时间片： 
4. GAT-LSTM-F(4)-L-Conv:block输出F(4)经过L变为F(1)，==卷积==取一个时间片： 
5. GAT-LSTM-F(1)-vcap(1)-LL:block输出F(1)与vcap(1) cat 并且取最后一个时间片的值: 
6. GAT-LSTM-F(4)-L-vcap(1)-LL:block输出F(4)经过L输出F(1)与vcap(1) cat 并且取最后一个时间片的值: 
7. GAT-LSTM-F(1)-vcap(1)-L-Conv:block输出F(1)与vcap(1) cat ==卷积==取一个时间片: 
8. GAT-LSTM-F(4)-L-vcap(1)-LL-Conv:block输出F(4)经过L输出F(1)与vcap(1) cat ==卷积==取一个时间片: 
10. GAT-LSTM-F(4)-vcap(1)-L-cat-LL-Conv:block输出F(4),vcap(1)经过L变成vcap(4) 之后cat LL ==卷积==取一个时间片:
11. GAT-LSTM-F(4)-vcap(1)-L-cat-LL:block输出F(4),vcap(1)经过L变成vcap(4) 之后cat LL 并且取最后一个时间片的值：
