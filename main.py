from simpletransformers.classification import ClassificationModel
from rdkit import Chem
import argparse

def seperatior_checker(rxn):
    if rxn.find('>>')!=-1:
        reactants = rxn.split('>>')[0]
        product = rxn.split('>>')[1]
        if Chem.MolFromSmiles(product)!= None:
            reactants_sep = reactants.split('.')
            for smiles in reactants_sep:
                if Chem.MolFromSmiles(smiles)!=None:
                    continue
                else:
                    raise SyntaxError(f'Invalid Smiles in reactants {smiles}')
        else:
            raise SyntaxError(f'Invalid Smiles in product {product}')

    return True

def parse_args():

    parser = argparse.ArgumentParser(description='Run Buchwald Hartwig Yield prediction from command line')
    parser.add_argument('-s', '--reaction', default=None, type = str, 
                        help='Reaction input for yield predictions')
    parser.add_argument('-n', '--name', default='test_reaction', help='The name of the molecule')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if bool(args.reaction) == True:
        if seperatior_checker(args.reaction) == True:
            model = ClassificationModel('roberta', 'Parsa/Buchwald-Hartwig-Yield-prediction',use_cuda=False, num_labels=1, args={
                                      "regression": True})
            pred, _ = model.predict([args.reaction])
            print(f'{abs(pred)*100} %')

        else:
            print('Invalid Reaction')
    else:
        print('Empty input')
