import numpy as np
import scipy
import scipy.stats
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt
import seaborn as sns

def read_nmr(filename, atom1_col, value_col, atom2_col=None):
    """
    Parse column in data files
    """
    with open(filename) as f:
        lines = f.readlines()


    data = {}
    for line in lines[1:]:
        tokens = line.split()
        mol = int(tokens[0])
        atom1 = int(tokens[atom1_col])
        value = float(tokens[value_col])
        if mol not in data: 
            data[mol] = {}
        if atom2_col is None:
            if atom1 not in data[mol]: data[mol][atom1] = 0
            data[mol][atom1] = value
        else:
            atom2 = int(tokens[atom2_col])
            pair = (atom1, atom2)
            if pair not in data[mol]: data[mol][pair] = 0
            data[mol][pair] = value
    return data

def concatenate(data):
    """
    Get all values from dicts of dicts of floats
    """
    def recursive_concatenate(x, concat_array=[]):
        if isinstance(x, float):
            concat_array.append(x)
        else:
            # dict
            sorted_keys = sorted(list(x.keys()))
            for i in sorted_keys:
                recursive_concatenate(x[i], concat_array)
        return concat_array

    return np.asarray(recursive_concatenate(data))

def find_duplicates(data):
    """
    Find indices of duplicate entries (e.g. methyles)
    """
    duplicates = {}
    for mol in data:
        keys = np.asarray(list(data[mol].keys()))
        labels = np.asarray(list(data[mol].values()))
        sort_idx = np.argsort(labels)
        sorted_labels = labels[sort_idx]
        sorted_keys = keys[sort_idx]

        duplicate_labels = []
        c = -1
        for i, label_i in enumerate(sorted_labels[:-1]):
            if i < c:
                continue
            if isinstance(sorted_keys[i], int):
                duplicate_labels_i = [sorted_keys[i]]
            else:
                duplicate_labels_i = [tuple(sorted_keys[i])]
            for j, label_j in enumerate(sorted_labels[i+1:]):
                c = j+i+1
                if label_j > label_i:
                    break
                elif label_j < label_i:
                    raise SystemExit("Should never get here")
                else:
                    if isinstance(sorted_keys[c], int):
                        duplicate_labels_i.append(sorted_keys[c])
                    else:
                        duplicate_labels_i.append(tuple(sorted_keys[c]))

            if len(duplicate_labels_i) > 1:
                duplicate_labels.append(duplicate_labels_i)

        if len(duplicate_labels) > 0:
            duplicates[mol] = duplicate_labels
    return duplicates

def remove_unique_entries(A,B):
    """
    Remove any entries that only exists in one of the data sets
    """
    A_common, B_common = {}, {}
    common_keys = list(set(A.keys()) & set(B.keys()))
    sorted_keys = sorted(common_keys)
    for key in sorted_keys:
        A_common[key] = {}
        B_common[key] = {}
        common_subkeys = list(set(A[key].keys()) & set(B[key].keys()))
        sorted_subkeys = sorted(common_subkeys)
        for subkey in sorted_subkeys:
            A_common[key][subkey] = A[key][subkey]
            B_common[key][subkey] = B[key][subkey]
    return A_common, B_common

def mean_duplicates(data, duplicates):
    data_unique = {}
    sorted_keys = sorted(list(data.keys()))
    for mol in sorted_keys:
        if mol in duplicates:
            data_unique[mol] = {}
            duplicate_keys = []
            sorted_subkeys = sorted(list(data[mol].keys()))
            for key1 in sorted_subkeys:
                if key1 in duplicates[mol]:
                    continue
                for unique_keys in duplicates[mol]:
                    if key1[0] == unique_keys[0][0] and key1[1] == unique_keys[0][1]:
                        duplicate_keys.extend(unique_keys)
                        label_sum = 0
                        for key2 in unique_keys:
                            label_sum += data[mol][key2]
                        data_unique[mol][key1] = label_sum / len(unique_keys)
                        break
                # key1 not in duplicate keys, doesn't work
                #if key1 not in duplicate_keys:
                if sum((key1[0] == key2[0]) and (key1[1] == key2[1]) for key2 in duplicate_keys) == 0:
                    data_unique[mol][key1] = data[mol][key1]
        else:
            data_unique[mol] = data[mol]
    return data_unique

class ProbabilityModel(object):
    """
    Class to handle fitting probability distributions
    """

    def __init__(self, distribution="t", scaling=False):
        if distribution not in ["cauchy", "normal", "laplace", "t"]:
            raise SystemExit("Unknown distribution: %s" % distribution)
        self.distribution = distribution
        self.scaling = scaling
        self.params = None
        self._variances_fitted = None
        self.bic = None
        self.aicc = None
        self.aic = None

    def _log_probability(self, x=None, scale=None, loc=None, df=None):
        if self.distribution == "cauchy":
            log_p = np.log(scale/np.pi) - np.log((x-loc)**2 + scale**2)
        elif self.distribution == "normal":
            log_p = - 0.5 * np.log(2 * np.pi * scale**2) - (x-loc)**2 / (2*scale**2)
        elif self.distribution == "laplace":
            log_p = - np.log(2 * scale) - abs(x-loc)/scale
        elif self.distribution == "t":
            log_p = np.log(scipy.special.gamma((df+1)/2) / (scipy.special.gamma(df/2)*np.sqrt(np.pi*df)*scale)) - (df+1)/2 * np.log(1+((x-loc)/scale)**2 / df)
        return log_p

    def _loss_function(self, x, labels, predictions, variances=None):
        """
        Loss function for the optimization step
        """

        n = len(labels)
        loc = x[0]
        if self._variances_fitted:
            a = np.exp(x[1])
            b = x[2]
        else:
            scale = np.exp(x[1])
        if self.distribution == "t":
            df = np.exp(x[2 + int(self._variances_fitted)])
        else:
            df = None
        if self.scaling:
            slope = x[-1]
        else:
            slope = 1

        negative_log_likelihood = 0
        for i in range(n):
            if self._variances_fitted:
                scale = a * variances[i]**b
            error = labels[i] - slope * predictions[i]
            negative_log_likelihood -= self._log_probability(error, scale, loc, df)

        return negative_log_likelihood

    def fit(self, labels, predictions, variances=None):
        """
        Fit probability distributions to the errors, optionally utilizing individual variances
        and fitting the scale parameter to follow a power-law
        """

        
        self._variances_fitted = True
        if variances is None:
            self._variances_fitted = False

        if self.scaling:
            slope, intercept = scipy.stats.linregress(predictions, labels)[:2]

        # Just make some good initial guesses, since the likelihood might not be convex
        if self.distribution == "normal":
            if self._variances_fitted:
                if self.scaling:
                    errors = labels - slope * predictions - intercept
                    # [loc, a, b, slope]
                    x0 = [intercept, np.log(np.std(errors)), 0, slope]
                else:
                    errors = labels - predictions
                    # [loc, a, b]
                    x0 = [np.mean(errors), np.log(np.std(errors)), 0]
            else:
                if self.scaling:
                    errors = labels - slope * predictions - intercept
                    # [loc, log_scale, slope]
                    x0 = [intercept, np.log(np.std(errors)), 1]
                else:
                    errors = labels - predictions
                    # [loc, log_scale]
                    x0 = [np.mean(errors), np.log(np.std(errors))]
        elif self.distribution == "cauchy":
            if self._variances_fitted:
                if self.scaling:
                    errors = labels - slope*predictions - intercept
                    sorted_errors = np.sort(errors)
                    n = sorted_errors.size
                    # 24% mid to do estimates
                    start = int(0.38 * n+ 1 )
                    end = int(0.62 * n + 1)
                    loc = np.mean(sorted_errors[start:end])
                    # half the interquartile range as estimator of scale
                    log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
                    # [loc, a, b, slope]
                    x0 = [intercept + loc, log_scale, 0, slope]
                else:
                    errors = labels -predictions 
                    sorted_errors = np.sort(errors)
                    n = sorted_errors.size
                    # 24% mid to do estimates
                    start = int(0.38 * n+ 1 )
                    end = int(0.62 * n + 1)
                    loc = np.mean(sorted_errors[start:end])
                    # half the interquartile range as estimator of scale
                    log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
                    # [loc, a, b]
                    x0 = [loc, log_scale, 0]
            else:
                if self.scaling:
                    errors = labels - slope*predictions - intercept
                    sorted_errors = np.sort(errors)
                    n = sorted_errors.size
                    # 24% mid to do estimates
                    start = int(0.38 * n+ 1 )
                    end = int(0.62 * n + 1)
                    loc = np.mean(sorted_errors[start:end])
                    # half the interquartile range as estimator of scale
                    log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
                    # [loc, log_scale, slope]
                    x0 = [intercept + loc, log_scale, slope]
                else:
                    errors = labels -predictions 
                    sorted_errors = np.sort(errors)
                    n = sorted_errors.size
                    # 24% mid to do estimates
                    start = int(0.38 * n+ 1 )
                    end = int(0.62 * n + 1)
                    loc = np.mean(sorted_errors[start:end])
                    # half the interquartile range as estimator of scale
                    log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
                    # [loc, log_scale]
                    x0 = [loc, log_scale]
        elif self.distribution == "t":
            if self._variances_fitted:
                if self.scaling:
                    errors = labels - slope*predictions - intercept
                    sorted_errors = np.sort(errors)
                    n = sorted_errors.size
                    # 24% mid to do estimates
                    start = int(0.38 * n+ 1 )
                    end = int(0.62 * n + 1)
                    loc = np.mean(sorted_errors[start:end])
                    # half the interquartile range as estimator of scale
                    log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
                    # [loc, a, b, log_df, slope]
                    x0 = [intercept + loc, log_scale, 0, 1, slope]
                else:
                    errors = labels -predictions 
                    sorted_errors = np.sort(errors)
                    n = sorted_errors.size
                    # 24% mid to do estimates
                    start = int(0.38 * n+ 1 )
                    end = int(0.62 * n + 1)
                    loc = np.mean(sorted_errors[start:end])
                    # half the interquartile range as estimator of scale
                    log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
                    # [loc, a, b, log_df]
                    x0 = [loc, log_scale, 0, 1]
            else:
                if self.scaling:
                    errors = labels - slope*predictions - intercept
                    sorted_errors = np.sort(errors)
                    n = sorted_errors.size
                    # 24% mid to do estimates
                    start = int(0.38 * n+ 1 )
                    end = int(0.62 * n + 1)
                    loc = np.mean(sorted_errors[start:end])
                    # half the interquartile range as estimator of scale
                    log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
                    # [loc, log_scale, log_df, slope]
                    x0 = [intercept + loc, log_scale, 1, slope]
                else:
                    errors = labels -predictions 
                    sorted_errors = np.sort(errors)
                    n = sorted_errors.size
                    # 24% mid to do estimates
                    start = int(0.38 * n+ 1 )
                    end = int(0.62 * n + 1)
                    loc = np.mean(sorted_errors[start:end])
                    # half the interquartile range as estimator of scale
                    log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
                    # [loc, log_scale, log_df]
                    x0 = [loc, log_scale, 1]
        elif self.distribution == "laplace":
            if self._variances_fitted:
                if self.scaling:
                    errors = labels - slope * predictions - intercept
                    # [loc, a, b, slope]
                    x0 = [intercept, np.log(np.mean(abs(errors))), 0, slope]
                else:
                    errors = labels - predictions
                    # [loc, a, b]
                    x0 = [np.median(errors), np.log(np.mean(abs(errors))), 0]
            else:
                if self.scaling:
                    errors = labels - slope * predictions - intercept
                    # [loc, log_scale, slope]
                    x0 = [intercept, np.log(np.mean(abs(errors))), 1]
                else:
                    errors = labels - predictions
                    # [loc, log_scale]
                    x0 = [np.median(errors), np.log(np.mean(abs(errors)))]

        res = scipy.optimize.minimize(self._loss_function, x0, args=(labels, predictions, variances), method="bfgs")
        self.params = res.x
        self.bic = np.log(len(errors)) * len(x0) + 2 * res.fun
        self.aic = 2 * len(x0) + 2 * res.fun
        self.aicc = self.aic + (2 * len(x0)**2 + 2 * len(x0)) / (len(errors) - len(x0) - 1)
        return self

    def get_log_likelihood(self, labels, predictions, variances=None):
        """
        Get the log-likelihood of some prediction errors and optionally variances
        """
        if self.params is None:
            raise SystemExit("Model has not been fitted")
        elif variances is None and self._variances_fitted:
            raise SystemExit("Variances has to be fitted to be able to used to get log-likelihood")
        elif variances is not None and not self._variances_fitted:
            raise SystemExit("Variances was fitted, so is also needed to get log-likelihood")

        if self.scaling:
            errors = labels - self.params[-1] * predictions
            end = -1
        else:
            errors = labels - predictions
            end = len(self.params)+1

        if self.distribution == "t":
            if self._variances_fitted:
                loc, log_a, b, log_df = self.params[:end]
                a = np.exp(log_a)
                df = np.exp(log_df)
                return sum(self._log_probability(error, scale=a*variance**b, df=df, loc=loc) for (error, variance) \
                        in zip(errors, variances))
            else:
                loc, log_scale, log_df = self.params[:end]
                scale = np.exp(log_scale)
                df = np.exp(log_df)
                return sum(self._log_probability(error, scale=scale, df=df, loc=loc) for error in errors)
        else:
            if self._variances_fitted:
                loc, log_a, b = self.params[:end]
                a = np.exp(log_a)
                return sum(self._log_probability(error, scale=a*variance**b, loc=loc) for (error, variance) \
                        in zip(errors, variances))
            else:
                loc, log_scale = self.params[:end]
                scale = np.exp(log_scale)
                return sum(self._log_probability(error, scale=scale, loc=loc) for error in errors)

def mae(x):
    """
    Return MAE of data
    """
    return np.mean(abs(x))

def rmse(x):
    """
    Return RMSE of data
    """
    return np.sqrt(np.mean(x**2))

def maxe(x):
    """
    Return MaxE of data
    """
    return np.max(abs(x))

def transform_data(data_exp, data_dft, data_ml, data_var, return_dict=False):
    """
    Do all kinds of tedious transformations to get array of values
    """
    _, data_dft_truncated = remove_unique_entries(data_exp, data_dft)
    _, data_ml_truncated = remove_unique_entries(data_exp, data_ml)
    _, data_var_truncated = remove_unique_entries(data_exp, data_var)

    data_exp_duplicates = find_duplicates(data_exp)

    data_exp_unique = mean_duplicates(data_exp, data_exp_duplicates)
    data_dft_unique = mean_duplicates(data_dft_truncated, data_exp_duplicates)
    data_ml_unique = mean_duplicates(data_ml_truncated, data_exp_duplicates)
    data_var_unique = mean_duplicates(data_var_truncated, data_exp_duplicates)

    if return_dict:
        for mol in data_exp_unique:
            data_exp_unique[mol] = concatenate(data_exp_unique[mol])
            data_dft_unique[mol] = concatenate(data_dft_unique[mol])
            data_ml_unique[mol] = concatenate(data_ml_unique[mol])
            data_var_unique[mol] = concatenate(data_var_unique[mol])
        return data_exp_unique, data_dft_unique, data_ml_unique, data_var_unique

    data_dft_ = concatenate(data_dft_unique)
    data_ml_ = concatenate(data_ml_unique)
    data_exp_ = concatenate(data_exp_unique)
    data_var_ = concatenate(data_var_unique)

    return data_exp_, data_dft_, data_ml_, data_var_

def J1CH_results():
    # Parsing data split up into more steps to make it easier to do changes
    syg_exp = read_nmr("__JUL07___SYG_exp_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=4)
    syg_dft = read_nmr("__JUL07___SYG_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=4)
    syg_ml = read_nmr("__JUL07___SYG_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=5)
    syg_var = read_nmr("__JUL07___SYG_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=6)
    # swapped atom1/atom2
    str_exp = read_nmr("STR_J1CH_val_raw.txt",atom1_col=2, atom2_col=1, value_col=3)
    str_dft = read_nmr("__JUL07___STRYCH_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=4)
    str_ml = read_nmr("__JUL07___STRYCH_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=5)
    str_var = read_nmr("__JUL07___STRYCH_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=6)

    # make copies of entries for str_exp, since the experimental values doesn't change per diastereomer in this case
    for mol in str_dft:
        str_exp[mol] = str_exp[1]

    syg_exp, syg_dft, syg_ml, syg_var = transform_data(syg_exp, syg_dft, syg_ml, syg_var)
    str_exp, str_dft, str_ml, str_var = transform_data(str_exp, str_dft, str_ml, str_var, True)

    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - no variance filter ***")
    errors = (syg_exp - 10.91 - syg_ml)
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_ml)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_ml)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - 10.91 - syg_ml
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_ml)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_ml)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - 10.91 - syg_dft
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 10.91 Hz offset - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_dft)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_dft)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 10.91 Hz offset - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - 10.91 - str_ml[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 10.91 - str_ml[mol])[str_var[mol] < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 10.91 - str_ml[mol])[str_var[mol] < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 10.91 Hz offset - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - 10.91 - str_dft[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 10.91 - str_dft[mol])[str_var[mol] < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 10.91 - str_dft[mol])[str_var[mol] < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))

    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 11 Hz offset - no variance filter ***")
    errors = (syg_exp - 11 - syg_ml)
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 11 - syg_ml)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 11 Hz offset - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 11 - syg_ml)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 11 Hz offset - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - 11 - syg_ml
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 11 Hz offset - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 11 - syg_ml)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 11 Hz offset - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 11 - syg_ml)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 11 Hz offset - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - 11 - syg_dft
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 11 Hz offset - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 11 - syg_dft)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 11 Hz offset - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 11 - syg_dft)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 11 Hz offset - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 11 Hz offset - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - 11 - str_ml[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 11 Hz offset - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 11 - str_ml[mol])[str_var[mol] < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 11 Hz offset - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 11 - str_ml[mol])[str_var[mol] < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 11 Hz offset - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - 11 - str_dft[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 11 Hz offset - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 11 - str_dft[mol])[str_var[mol] < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 11 Hz offset - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 11 - str_dft[mol])[str_var[mol] < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))

    for dist in ("laplace", "normal", "cauchy", "t"):
        for scaling in True, False:
            ml_model = ProbabilityModel(distribution=dist, scaling=scaling)
            ml_model.fit(syg_exp, syg_ml, syg_var)
            dft_model = ProbabilityModel(distribution=dist, scaling=scaling)
            dft_model.fit(syg_exp, syg_dft)
            print(dist, scaling, ml_model.bic, dft_model.bic, ml_model.aicc, dft_model.aicc)
    ml_model = ProbabilityModel(distribution='laplace', scaling=False)
    ml_model.fit(syg_exp, syg_ml, syg_var)
    dft_model = ProbabilityModel(distribution='laplace', scaling=False)
    dft_model.fit(syg_exp, syg_dft)

    # Print all the things
    print("*** Parameters - Sygenta - ML to exp ***")
    print(ml_model.params)
    print("*** Parameters - Sygenta - DFT to exp ***")
    print(dft_model.params)

    ml_ll = []
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - ML to exp ***")
    ll1 = ml_model.get_log_likelihood(str_exp[1], str_ml[1], str_var[1])
    for mol in str_dft.keys():
        ll = ml_model.get_log_likelihood(str_exp[1], str_ml[mol], str_var[mol])
        ml_ll.append(ll)
        print(mol, ll, np.exp(ll1-ll))

    dft_ll = []
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - DFT to exp ***")
    ll1 = dft_model.get_log_likelihood(str_exp[1], str_dft[1])
    for mol in str_dft.keys():
        ll = dft_model.get_log_likelihood(str_exp[1], str_dft[mol])
        dft_ll.append(ll)
        print(mol, ll, np.exp(ll1-ll))

    return np.asarray(ml_ll), np.asarray(dft_ll)


if __name__ == "__main__":
    J1CH_results()
    quit()
    H1_results()
    C13_results()
    quit()
    labels, predictions, variances = read_nmr("__JUL03___CD_exp_CCS_raw.txt", -1)
    P = ProbabilityModel(distribution='laplace', scaling=True)
    P.fit(labels, predictions, variances)
    print(P.params, P.bic, P.aicc)
    labels, predictions, variances = read_nmr("__JUL03___CD_exp_HCS_raw.txt", -1)
    P = ProbabilityModel(distribution='laplace', scaling=True)
    P.fit(labels, predictions, variances)
    print(P.params, P.bic, P.aicc)
    quit()




