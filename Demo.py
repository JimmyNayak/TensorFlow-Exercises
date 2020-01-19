from sklearn import tree

productFeature = [[140, 1], [130, 1], [150, 0], [170, 0]]
productLabels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(productFeature, productLabels)

print(clf.predict([[160, 0]]))
