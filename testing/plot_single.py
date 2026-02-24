import matplotlib.pyplot as plt


x, y, p = [], [], [];


with open("./output/out.txt", "r") as file:
    x = list(map(float,file.readline().split()));
    y = list(map(float,file.readline().split()));
    p = list(map(float,file.readline().split()));


plt.plot(x, y, color = "red", label = "Expected output");
plt.plot(x, p, color = "blue", label = "Network output");
plt.title("Network learning output");
plt.legend(loc = "best");
plt.xlabel("x");
plt.ylabel("y");
plt.show();
