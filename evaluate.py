import argparse as ap

def main(txt):
    '''row by row entity evaluation: we evaluate by whole named entities'''
    tp = 0; fp = 0; fn = 0
    in_entity = 0
    for i in txt.index:
      if txt['prediction'][i]=='B' and txt['bio_only'][i]=='B':
        if in_entity==1:  # if there's a preceding named entity which didn't have intervening O...
          tp += 1  # count a true positive
        in_entity = 1  # start tracking this entity (don't count it until we know full span of entity)
      elif txt['prediction'][i]=='B':
        fp += 1  # if not a B in gold annotations, it's a false positive
        in_entity = 0
      elif txt['prediction'][i]=='I' and txt['bio_only'][i]=='I':
        next  # correct entity continuation: do nothing
      elif txt['prediction'][i]=='I' and txt['bio_only'][i]=='B':
        fn += 1  # if a new entity should have begun, it's a false negative
        in_entity = 0
      elif txt['prediction'][i]=='I':  # if gold is O...
        if in_entity==1:  # and if tracking an entity, then the span is too long
          fp += 1  # it's a false positive
        in_entity = 0
      elif txt['prediction'][i]=='O':
        if txt['bio_only'][i]=='B':
          fn += 1  # false negative if there's B in gold but no predicted B
          if in_entity==1:  # also check if there was a named entity in progress
            tp += 1  # count a true positive
        elif txt['bio_only'][i]=='I':
          if in_entity==1:  # if this should have been a continued named entity, the span is too short
            fn += 1  # count a false negative
        elif txt['bio_only'][i]=='O':
          if in_entity==1:  # if a named entity has ended in right place
            tp += 1  # count a true positive
        in_entity = 0

    if in_entity==1:  # catch any final named entity
      tp += 1

    print('Sum of TP and FP = %i' % (tp+fp))
    print('Sum of TP and FN = %i' % (tp+fn))
    print('True positives = %i, False positives = %i, False negatives = %i' % (tp, fp, fn))
    prec = tp / (tp+fp)
    rec = tp / (tp+fn)
    f1 = (2*(prec*rec)) / (prec+rec)
    print('Precision = %.3f, Recall = %.3f, F1 = %.3f (max=1)' % (prec, rec, f1))

if __name__ == "__main__":
    
    p = ap.ArgumentParser()
    p.add_argument('preds', required=True, \
        help='txt file of predictions to evaluate')
    args = p.parse_args()
    
    main(args.preds)