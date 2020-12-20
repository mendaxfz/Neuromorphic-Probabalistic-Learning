class Scheduler:
    """Takes in the vector of values and returns a value as a function of epoch number.
        _get_val method returns a value from the vector at an index 'epoch'
        Note: if an index is out of bound then the last element of val_vector is returned
    """
    def __init__(self,val_vector):
        if not (type(val_vector) is list):
            val_vector = list(val_vector)
        self.val_vector = val_vector

    def _get_val(self,epoch):
        indx = epoch
        if epoch >= len(self.val_vector):
            indx = -1
        return self.val_vector[indx]
