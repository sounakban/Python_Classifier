class getRCV1V2:

    def __init__(self, path, testset = 1):
        if (testset < 1 or testset > 4):
            raise ValueError('Valid values of set are 1,2,3,4. Enter Valid value.')
        self.trainfile = "{}{}".format(path, "TrainingData/lyrl2004_tokens_train.dat")
        open(self.trainfile, 'r')
        self.testfile = "{}{}{}{}".format(path, "TestData/lyrl2004_tokens_test_pt", testset-1, ".dat")
        open(self.trainfile, 'r')
        self.readFiles()

    def readFiles(self, ):
        self.data = []
        self.docids = []

        currDoc = ""
        for line in open(self.trainfile, 'r'):
            item = line.rstrip()
            if item.startswith(".W") or len(item)==0:
                continue
            elif item.startswith(".I"):
                if len(currDoc)!=0:
                    self.data.append(currDoc)
                    self.docids.append(currID)
                    currDoc = ""
                currID = int(item.split()[1])
            else:
                currDoc = "{}{}. ".format(currDoc, item)
        self.data.append(currDoc)
        self.docids.append(currID)
        self.trainDocCount = len(self.data)

        currDoc = ""
        for line in open(self.testfile, 'r'):
            item = line.rstrip()
            if item.startswith(".W") or len(item)==0:
                continue
            elif item.startswith(".I"):
                if len(currDoc)!=0:
                    self.data.append(currDoc)
                    self.docids.append(currID)
                    currDoc = ""
                currID = int(item.split()[1])
            else:
                currDoc = "{}{}. ".format(currDoc, item)
        self.data.append(currDoc)
        self.docids.append(currID)
        self.totalDocCount = len(self.data)



    def getData(self):
        return self.data

    def getDocIDs(self):
        return self.docids

    def getTrainDocCount(self):
        return self.trainDocCount

    def getTotalDocCount(self):
        return self.totalDocCount
