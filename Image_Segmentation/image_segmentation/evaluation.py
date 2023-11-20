import torch

# pred : Segmentation Result
# gt : Ground Truth


class EvaluationMatrix():
    def __init__(self):
        self.total_acc = 0.
        self.total_SE = 0.
        self.total_SP = 0.
        self.total_PC = 0.
        self.total_F1 = 0.
        self.total_JS = 0.
        self.total_DC = 0.

        self.acc = 0.       # Accuracy
        self.SE = 0.		# Sensitivity (Recall)
        self.SP = 0.		# Specificity
        self.PC = 0. 	    # Precision
        self.F1 = 0.		# F1 Score
        self.JS = 0.		# Jaccard Similarity
        self.DC = 0.		# Dice Coefficient
        self.length = 0

    def update(self, pred, gt):
        gt = gt == torch.max(gt)

        self.total_acc += self.get_accuracy(pred, gt)*gt.size(0)
        self.total_SE += self.get_sensitivity(pred, gt)*gt.size(0)
        self.total_SP += self.get_specificity(pred, gt)*gt.size(0)
        self.total_PC += self.get_precision(pred, gt)*gt.size(0)
        self.total_F1 += self.get_F1(pred, gt)*gt.size(0)
        self.total_JS += self.get_JS(pred, gt)*gt.size(0)
        self.total_DC += self.get_DC(pred, gt)*gt.size(0)
        self.length += gt.size(0)

    def update_print(self, sentence=''):
        self.acc = self.total_acc/self.length
        self.SE = self.total_SE / self.length
        self.SP = self.total_SP / self.length
        self.PC = self.total_PC / self.length
        self.F1 = self.total_F1 / self.length
        self.JS = self.total_JS / self.length
        self.DC = self.total_DC / self.length
        print(sentence+'Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
            self.acc, self.SE, self.SP, self.PC, self.F1, self.JS, self.DC))

    def get_accuracy(self, pred, gt):

        corr = torch.sum(pred == gt)
        tensor_size = pred.size(0)*pred.size(1)*pred.size(2)*pred.size(3)
        acc = float(corr)/float(tensor_size)

        return acc

    def get_sensitivity(self, pred, gt):
        # Sensitivity == Recall

        TP = ((pred == 1) & (gt == 1)).to(torch.int)  # ==2
        FN = ((pred == 0) & (gt == 1)).to(torch.int)  # ==2

        SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)

        return SE

    def get_specificity(self, pred, gt):

        # TN : True Negative
        # FP : False Positive
        # TN = ((pred==0)+(gt==0))==2
        # FP = ((pred==1)+(gt==0))==2
        TN = ((pred == 0) & (gt == 0)).to(torch.int)  # ==2
        FP = ((pred == 1) & (gt == 0)).to(torch.int)  # ==2

        SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)

        return SP

    def get_precision(self, pred, gt):

        # TP : True Positive
        # FP : False Positive
        # TP = ((pred==1)+(gt==1))==2
        # FP = ((pred==1)+(gt==0))==2
        TP = ((pred == 1) & (gt == 1)).to(torch.int)  # ==2
        FP = ((pred == 1) & (gt == 0)).to(torch.int)  # ==2

        PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

        return PC

    def get_F1(self, pred, gt):
        # Sensitivity == Recall
        SE = self.get_sensitivity(pred, gt)
        PC = self.get_precision(pred, gt)

        F1 = 2*SE*PC/(SE+PC + 1e-6)

        return F1

    def get_JS(self, pred, gt):
        # JS : Jaccard similarity

        pred = pred.to(torch.int)
        gt = gt.to(torch.int)
        Inter = torch.sum((pred+gt) == 2)
        Union = torch.sum((pred+gt) >= 1)

        JS = float(Inter)/(float(Union) + 1e-6)

        return JS

    def get_DC(self, pred, gt):
        # DC : Dice Coefficient

        pred = pred.to(torch.int)
        gt = gt.to(torch.int)

        Inter = torch.sum((pred+gt) == 2)
        DC = float(2*Inter)/(float(torch.sum(pred)+torch.sum(gt)) + 1e-6)

        return DC
