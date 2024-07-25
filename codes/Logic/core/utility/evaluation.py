
from typing import List
import wandb
import math
import os

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]],A) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        
        ll=len(predicted)
        for i in range(ll):
            p=0
            l=len(predicted[i])
            for j in predicted[i]:
                if j in actual[i]:
                    p=p+1

            precision=precision+(p/l)

        return precision/ll

            
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]],A) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        ll=len(predicted)
        for i in range(ll):
            r=0
            for j in predicted[i]:
                if j in actual[i]:
                    r=r+1

            recall=recall+(r/len(actual[i]))

        return recall/ll
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]],A) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """

        p=self.calculate_precision(actual,predicted,A)
        r=self.calculate_recall(actual,predicted,A)
        f1 = 0.0

        f1=2*(p*r)/(p+r)

        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]], i ,A) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0


        P=predicted[i]
        R=actual[i]

        T = [1 if p in R else 0 for p in P]

        s=0


        for t in range(len(T)):
            if T[t]==1:
                s=s+1
                AP=AP+(sum(T[:(t+1)])/(t+1))
        AP=AP/s

        return AP
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]],A) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        for i in range(len(predicted)):
            MAP=MAP+self.calculate_AP(actual,predicted,i,A)
        
        MAP=MAP/(len(predicted))
            

        return MAP
    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]],A) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0.0


        for i in range(len(predicted)):
            PRED=predicted[i]
            REL=A[i]
            n = len(PRED)
            total_sum = 0
            for j in range(n):
                total_sum += (2**(REL[j]) - 1) / math.log2(j+2)
            DCG=DCG+total_sum

        

        return DCG/len(predicted)
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]],A) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        for i in range(len(predicted)):
            PRED=predicted[i]
            REL=A[i]
            #print(REL)
            n = len(PRED)
            total_sum = 0
            for J in range(1,n+1):
                total_sum += (2**(REL[J-1]) - 1) / math.log2(J+1)

            newRel=sorted(REL,reverse=True)

            #print(newRel)

            new_total_sum = 0

            for J in range(1,n+1):
                new_total_sum += (2**(newRel[J-1]) - 1) / math.log2(J+1)









            NDCG=NDCG+(total_sum/new_total_sum)


        return NDCG/len(predicted)
    
    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]],i,A) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = 0.0

        pred=predicted[i]
        rel=actual[i]

        r=0

        for j in range(len(pred)):
            if pred[j] in rel:
                r=j+1
                break


        return (1/r)

    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]],A) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        for i in range(len(predicted)):
            MRR=MRR+self.cacluate_RR(actual,predicted,i,A)

        return MRR/(len(predicted))
    
    #def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
    def print_evaluation(self, precision, recall, f1, map, dcg, ndcg, mrr,A):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        print("Evaluation Metrics:")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        #print(f"Average Precision: {ap}")
        print(f"Mean Average Precision: {map}")
        print(f"Discounted Cumulative Gain: {dcg}")
        print(f"Normalized Discounted Cumulative Gain: {ndcg}")
        #print(f"Reciprocal Rank: {rr}")
        print(f"Mean Reciprocal Rank: {mrr}")

      
    def log_evaluation(self, precision, recall, f1, map, dcg, ndcg, mrr,A):
    #def log_evaluation(self, precision, recall, f1, map, mrr,A,B):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """

        os.environ['WANDB_API_KEY'] = '160e360621831043156e8e52f3e22b02f9d93128'

        wandb.login()



        wandb.init(
    
            project="my-awesome-project",

        )
        

        wandb.log({"Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                #"Average Precision": ap,
                "Mean Average Precision": map,
                "Discounted Cumulative Gain": dcg,
                "Normalized Discounted Cumulative Gain": ndcg,
                #"Reciprocal Rank": rr,
                "Mean Reciprocal Rank": mrr})

    
        wandb.finish()


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]],A):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted,A)
        recall = self.calculate_recall(actual, predicted,A)
        f1 = self.calculate_F1(actual, predicted,A)
        #ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted,A)
        dcg = self.cacluate_DCG(actual, predicted,A)
        ndcg = self.cacluate_NDCG(actual, predicted,A)
        #rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted,A)

        #call print and viualize functions



        #self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        #self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)

        self.print_evaluation(precision, recall, f1, map_score,dcg,ndcg ,mrr,A)
        self.log_evaluation(precision, recall, f1, map_score,dcg,ndcg, mrr,A)

if __name__ == "__main__":
    eval = Evaluation('test')
    eval.calculate_evaluation([['a','b']],[['b','c']],[[2,6]])
    