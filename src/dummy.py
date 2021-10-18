titles = ["a", "b", "c"]
y_test = [
    ["1", "2", "3"],
    ["4", "5", "6"],
    ["9", "8", "7"],
]
y_plot = []
for i in titles:
    y_plot.append([[], []])

for point in y_test:
    for i in range(len(titles)):
        y_plot[i][0].append(point[i])

print(y_plot)
