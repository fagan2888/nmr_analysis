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

            if isinstance(sorted_keys[i], (int, np.int64)):
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
                    if isinstance(sorted_keys[c], (int, np.int64)):
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
                    flag = False
                    if isinstance(key1, (int, np.int64)):
                        if key1 == unique_keys[0]:
                            flag = True
                    elif key1[0] == unique_keys[0][0] and key1[1] == unique_keys[0][1]:
                        flag = True
                    if flag:
                        duplicate_keys.extend(unique_keys)
                        label_sum = 0
                        for key2 in unique_keys:
                            label_sum += data[mol][key2]
                        data_unique[mol][key1] = label_sum / len(unique_keys)
                        break
                # key1 not in duplicate keys, doesn't work
                #if key1 not in duplicate_keys:
                if isinstance(key1, (int, np.int64)):
                    if sum(key1 == key2 for key2 in duplicate_keys) == 0:
                        data_unique[mol][key1] = data[mol][key1]
                elif sum((key1[0] == key2[0]) and (key1[1] == key2[1]) for key2 in duplicate_keys) == 0:
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


    def _get_loglikelihood(self, x, loc, scale, df=None):
        """
        Because scipy stats is slooow
        """
        if self.distribution == "cauchy":
            ll = np.log(scale/np.pi) - np.log((x-loc)**2 + scale**2)
        elif self.distribution == "normal":
            ll = - 0.5 * np.log(2 * np.pi * scale**2) - (x-loc)**2 / (2*scale**2)
        elif self.distribution == "laplace":
            ll = - np.log(2 * scale) - abs(x-loc)/scale
        elif self.distribution == "t":
            ll = np.log(scipy.special.gamma((df+1)/2) / (scipy.special.gamma(df/2)*np.sqrt(np.pi*df)*scale)) - (df+1)/2 * np.log(1+((x-loc)/scale)**2 / df)
        return ll

    def get_sample_likelihoods(self, variance=None, n=int(1e6)):
        """
        Get sampled likelihoods from the fitted distribution. Done mostly analytically to avoid slow scipy.
        """

        if self.params is None:
            raise SystemExit("Model has not been fitted")
        elif variance is None and self._variances_fitted:
            raise SystemExit("Variances has to be fitted to be able to used to get log-likelihood")
        elif variance is not None and not self._variances_fitted:
            raise SystemExit("Variances was fitted, so is also needed to get log-likelihood")

        if self._variances_fitted:
            a = np.exp(self.params[1])
            b = self.params[2]
            scale = a * variance**b
        else:
            scale = np.exp(self.params[1])
        loc = self.params[0]

        p = np.random.random(n)
        if self.distribution == 'laplace':
            ll = np.log(np.minimum(p,1-p)) - np.log(scale)
        elif self.distribution == 'cauchy':
            ll = np.log(np.sin(p*np.pi)**2)-np.log(np.pi*scale)
        elif self.distribution == 'normal':
            ll = -np.log(np.sqrt(2*np.pi)*scale) - scipy.special.erfinv(2*p-1)**2
        elif self.distribution == "t":
            df = np.exp(self.params[2 + int(self._variances_fitted)])
            # Can't solve the equations
            dist = scipy.stats.t(df=df, loc=loc, scale=scale)
            samples = dist.ppf(p)
            ll = dist.logpdf(samples)

        return ll


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

        errors = labels - slope * predictions

        if self._variances_fitted:
            log_likelihood = 0
            for i in range(n):
                scale = a * variances[i]**b
                log_likelihood += self._get_loglikelihood(errors[i], loc, scale, df)
        else:
            log_likelihood = np.sum(self._get_loglikelihood(errors, loc, scale, df))

        return -log_likelihood

    def fit(self, labels, predictions, variances=None):
        """
        Fit probability distributions to the errors, optionally utilizing individual variances
        and fitting the scale parameter to follow a power-law
        """

        self._variances_fitted = True
        if variances is None:
            self._variances_fitted = False
        else:
            # Due to accuracy issues in the data
            variances_copy = variances.copy()
            variances_copy[variances < 5e-5] = 5e-5
            variances = variances_copy

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
        self.bic = np.log(len(errors)) * len(self.params) + 2 * res.fun
        self.aic = 2 * len(self.params) + 2 * res.fun
        self.aicc = self.aic + (2 * len(self.params)**2 + 2 * len(self.params)) / (len(errors) - len(self.params) - 1)
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

        negative_log_likelihood = self._loss_function(self.params, labels, predictions, variances)

        return -negative_log_likelihood



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

def J1CH_results(distribution='laplace'):
    # Parsing data split up into more steps to make it easier to do changes
    syg_exp = read_nmr("__JUL07___SYG_exp_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=4)
    syg_dft = read_nmr("__JUL07___SYG_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=4)
    syg_ml = read_nmr("__JUL07___SYG_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=5)
    syg_var = read_nmr("__JUL07___SYG_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=6)
    # swapped atom1/atom2
    str_exp = read_nmr("STR_J1CH_val_raw.txt", atom1_col=2, atom2_col=1, value_col=3)
    str_dft = read_nmr("__JUL07___STRYCH_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=4)
    str_ml = read_nmr("__JUL07___STRYCH_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=5)
    str_var = read_nmr("__JUL07___STRYCH_dft_1JCH_raw.txt", atom1_col=2, atom2_col=3, value_col=6)



    # make copies of entries for str_exp, since the experimental values doesn't change per diastereomer in this case
    for mol in str_dft:
        str_exp[mol] = str_exp[1]

    syg_exp, syg_dft, syg_ml, syg_var = transform_data(syg_exp, syg_dft, syg_ml, syg_var)
    str_exp, str_dft, str_ml, str_var = transform_data(str_exp, str_dft, str_ml, str_var, True)


    #x, y = [], []
    #for i in range(len(str_dft[mol])):
    #    for mol in str_dft:
    #        x.append(i)
    #        y.append(str_exp[1][i] - str_dft[mol][i])
    #plt.scatter(x, y)
    #plt.show()

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

    ml_model = ProbabilityModel(distribution=distribution, scaling=False)
    ml_model.fit(syg_exp, syg_ml, syg_var)
    dft_model = ProbabilityModel(distribution=distribution, scaling=False)
    dft_model.fit(syg_exp, syg_dft)

    # Print all the things
    print("*** Parameters/aicc - Sygenta - ML to exp ***")
    print(ml_model.params, ml_model.aicc)
    print("*** Parameters/aicc - Sygenta - DFT to exp ***")
    print(dft_model.params, dft_model.aicc)


    ml_ll = {}
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - ML to exp ***")
    ll1 = ml_model.get_log_likelihood(str_exp[1], str_ml[1], str_var[1])
    for mol in str_dft.keys():
        ll = ml_model.get_log_likelihood(str_exp[1], str_ml[mol], str_var[mol])
        ml_ll[mol] = ll
        print(mol, ll, np.exp(ll1-ll))

    dft_ll = {}
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - DFT to exp ***")
    ll1 = dft_model.get_log_likelihood(str_exp[1], str_dft[1])
    for mol in str_dft.keys():
        ll = dft_model.get_log_likelihood(str_exp[1], str_dft[mol])
        dft_ll[mol] = ll
        print(mol, ll, np.exp(ll1-ll))

    return ml_model, dft_model, str_exp, str_dft, str_ml, str_var

def H1_results(distribution='laplace'):
    # Parsing data split up into more steps to make it easier to do changes
    syg_exp = read_nmr("__JUL07___CD_exp_HCS_raw.txt", atom1_col=2, value_col=3)
    syg_dft = read_nmr("__JUL07___CD_dft_HCS_raw.txt", atom1_col=2, value_col=3)
    syg_ml = read_nmr("__JUL07___CD_dft_HCS_raw.txt", atom1_col=2, value_col=4)
    syg_var = read_nmr("__JUL07___CD_dft_HCS_raw.txt", atom1_col=2, value_col=5)
    # swapped atom1/atom2
    str_exp = read_nmr("STR_HCS_val_raw.txt",atom1_col=1, value_col=2)
    str_dft = read_nmr("__JUL07___STRYCH_dft_HCS_raw.txt", atom1_col=2, value_col=3)
    str_ml = read_nmr("__JUL07___STRYCH_dft_HCS_raw.txt", atom1_col=2, value_col=4)
    str_var = read_nmr("__JUL07___STRYCH_dft_HCS_raw.txt", atom1_col=2, value_col=5)

    # make copies of entries for str_exp, since the experimental values doesn't change per diastereomer in this case
    for mol in str_dft:
        str_exp[mol] = str_exp[1]

    syg_exp, syg_dft, syg_ml, syg_var = transform_data(syg_exp, syg_dft, syg_ml, syg_var)
    str_exp, str_dft, str_ml, str_var = transform_data(str_exp, str_dft, str_ml, str_var, True)

    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - no variance filter ***")
    errors = (syg_exp - syg_ml)
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_ml)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_ml)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - syg_ml
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_ml)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_ml)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - syg_dft
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - linear correction - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_dft)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - linear correction - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_dft)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - linear correction - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - linear correction - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - str_ml[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - linear correction - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - str_ml[mol])[str_var[mol] < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - linear correction - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - str_ml[mol])[str_var[mol] < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - linear correction - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - str_dft[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - linear correction - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - str_dft[mol])[str_var[mol] < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - linear correction - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - str_dft[mol])[str_var[mol] < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))


    ml_model = ProbabilityModel(distribution=distribution, scaling=True)
    ml_model.fit(syg_exp, syg_ml, syg_var)
    dft_model = ProbabilityModel(distribution=distribution, scaling=True)
    dft_model.fit(syg_exp, syg_dft)

    # Print all the things
    print("*** Parameters/aicc - CD - ML to exp ***")
    print(ml_model.params, ml_model.aicc)
    print("*** Parameters/aicc - CD - DFT to exp ***")
    print(dft_model.params, dft_model.aicc)

    ml_ll = {}
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - ML to exp ***")
    ll1 = ml_model.get_log_likelihood(str_exp[1], str_ml[1], str_var[1])
    for mol in str_dft.keys():
        ll = ml_model.get_log_likelihood(str_exp[1], str_ml[mol], str_var[mol])
        ml_ll[mol] = ll
        print(mol, ll, np.exp(ll1-ll))

    dft_ll = {}
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - DFT to exp ***")
    ll1 = dft_model.get_log_likelihood(str_exp[1], str_dft[1])
    for mol in str_dft.keys():
        ll = dft_model.get_log_likelihood(str_exp[1], str_dft[mol])
        dft_ll[mol] = ll
        print(mol, ll, np.exp(ll1-ll))

    return ml_model, dft_model, str_exp, str_dft, str_ml, str_var

def C13_results(distribution='laplace'):
    # Parsing data split up into more steps to make it easier to do changes
    syg_exp = read_nmr("__JUL07___CD_exp_CCS_raw.txt", atom1_col=2, value_col=3)
    syg_dft = read_nmr("__JUL07___CD_dft_CCS_raw.txt", atom1_col=2, value_col=3)
    syg_ml = read_nmr("__JUL07___CD_dft_CCS_raw.txt", atom1_col=2, value_col=4)
    syg_var = read_nmr("__JUL07___CD_dft_CCS_raw.txt", atom1_col=2, value_col=5)
    # swapped atom1/atom2
    str_exp = read_nmr("STR_CCS_val_raw.txt",atom1_col=1, value_col=2)
    str_dft = read_nmr("__JUL07___STRYCH_dft_CCS_raw.txt", atom1_col=2, value_col=3)
    str_ml = read_nmr("__JUL07___STRYCH_dft_CCS_raw.txt", atom1_col=2, value_col=4)
    str_var = read_nmr("__JUL07___STRYCH_dft_CCS_raw.txt", atom1_col=2, value_col=5)

    # make copies of entries for str_exp, since the experimental values doesn't change per diastereomer in this case
    for mol in str_dft:
        str_exp[mol] = str_exp[1]

    syg_exp, syg_dft, syg_ml, syg_var = transform_data(syg_exp, syg_dft, syg_ml, syg_var)
    str_exp, str_dft, str_ml, str_var = transform_data(str_exp, str_dft, str_ml, str_var, True)


    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - no variance filter ***")
    errors = (syg_exp - syg_ml)
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_ml)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_ml)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - syg_ml
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_ml)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_ml)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - linear correction - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - syg_dft
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - linear correction - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_dft)[syg_var < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - linear correction - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - syg_dft)[syg_var < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - linear correction - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - linear correction - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - str_ml[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - linear correction - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - str_ml[mol])[str_var[mol] < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - linear correction - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - str_ml[mol])[str_var[mol] < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - linear correction - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - str_dft[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - linear correction - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - str_dft[mol])[str_var[mol] < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - linear correction - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - str_dft[mol])[str_var[mol] < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))


    ml_model = ProbabilityModel(distribution=distribution, scaling=True)
    ml_model.fit(syg_exp, syg_ml, syg_var)
    dft_model = ProbabilityModel(distribution=distribution, scaling=True)
    dft_model.fit(syg_exp, syg_dft)

    # Print all the things
    print("*** Parameters/aicc - CD - ML to exp ***")
    print(ml_model.params, ml_model.aicc)
    print("*** Parameters/aicc - CD - DFT to exp ***")
    print(dft_model.params, dft_model.aicc)

    ml_ll = {}
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - ML to exp ***")
    ll1 = ml_model.get_log_likelihood(str_exp[1], str_ml[1], str_var[1])
    for mol in str_dft.keys():
        ll = ml_model.get_log_likelihood(str_exp[1], str_ml[mol], str_var[mol])
        ml_ll[mol] = ll
        print(mol, ll, np.exp(ll1-ll))

    dft_ll = {}
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - DFT to exp ***")
    ll1 = dft_model.get_log_likelihood(str_exp[1], str_dft[1])
    for mol in str_dft.keys():
        ll = dft_model.get_log_likelihood(str_exp[1], str_dft[mol])
        dft_ll[mol] = ll
        print(mol, ll, np.exp(ll1-ll))

    return ml_model, dft_model, str_exp, str_dft, str_ml, str_var

#def get_probability_of_true_models(ml_models, dft_models, str_exps, str_dfts, str_mls, str_vars):
#    """
#    Get the probabilities of a model actually being the best one and not just a fluke.
#    This is done numerically, by assuming that the true model follows the probability
#    distribution fitted from the experimental data. A probability distribution is fitted to
#    wrong data, and likelihoods are subsampled by random.
#    """
#
#    # The molecule that is considered the true one
#    for mol1 in str_dfts[0]:
#        ml_lls = []
#        dft_lls = []
#        wrong_ml_lls = []
#        wrong_dft_lls = []
#        for i in range(len(ml_models)):
#            # Get likelihoods for the model that we assume is the true one
#            ml_ll = ml_models[i].get_log_likelihood(str_exps[i][1], str_mls[i][mol1], str_vars[i][mol1])
#            dft_ll = dft_models[i].get_log_likelihood(str_exps[i][1], str_dfts[i][mol1])
#            ml_lls.append(ml_ll)
#            dft_lls.append(dft_ll)
#
#            # Fit models to the data we assume is wrong.
#            labels = np.concatenate([values for key, values in str_exps[i].items() if key != mol1])
#            dft_predictions = np.concatenate([values for key, values in str_dfts[i].items() if key != mol1])
#            ml_predictions = np.concatenate([values for key, values in str_mls[i].items() if key != mol1])
#            var = np.concatenate([values for key, values in str_vars[i].items() if key != mol1])
#
#            ## Use wrong_dists
#            #wrong_ml_model = ProbabilityModel(distribution=ml_models[i].distribution, scaling=ml_models[i].scaling)
#            #wrong_ml_model.fit(labels, ml_predictions, var)
#            #wrong_dft_model = ProbabilityModel(distribution=dft_models[i].distribution, scaling=dft_models[i].scaling)
#            #wrong_dft_model.fit(labels, dft_predictions)
#
#            # Slow
#            import time
#            t = time.time()
#            lls = []
#            c = 0
#            for j in range(100000):
#                wrong_ml_lls.append([])
#                wrong_dft_lls.append([])
#                for mol2 in str_dfts[0]:
#                    if mol2 == mol1:
#                        continue
#                    # Subsample points
#                    ## Leave out the "True" model
#                    #submols = np.random.choice([mol for mol in str_dfts[0] if mol != mol1], size=len(str_dfts[0][mol1]))
#                    submols = np.random.choice([mol for mol in str_dfts[0]], size=len(str_dfts[0][mol1]))
#                    ml_predictions = np.asarray([str_mls[i][submol][k] for k, submol in enumerate(submols)])
#                    dft_predictions = np.asarray([str_mls[i][submol][k] for k, submol in enumerate(submols)])
#                    var = np.asarray([str_vars[i][submol][k] for k, submol in enumerate(submols)])
#                    ## use wrong_dists
#                    #wrong_ml_ll = wrong_ml_model.get_log_likelihood(str_exps[i][1], ml_predictions, var)
#                    #wrong_dft_ll = wrong_dft_model.get_log_likelihood(str_exps[i][1], dft_predictions)
#                    wrong_ml_ll = ml_models[i].get_log_likelihood(str_exps[i][1], ml_predictions, var)
#                    wrong_dft_ll = dft_models[i].get_log_likelihood(str_exps[i][1], dft_predictions)
#                    wrong_ml_lls[-1].append(wrong_ml_ll)
#                    wrong_dft_lls[-1].append(wrong_dft_ll)
#                if any(dft_ll < np.asarray(wrong_dft_lls[-1])):
#                    c += 1
#            print(np.max(np.concatenate(wrong_ml_lls)))
#            print(c)
#            print(time.time() - t)
#            quit()


def get_probability_of_true_models(ml_models, dft_models, str_exps, str_dfts, str_mls, str_vars, n=int(1e6)):
    """
    Get the probabilities of a model actually being the best one and not just a fluke.
    This is done numerically, by counting how often two similar models would result in the
    same likelihood ratio as observed. 
    """

    n_models = len(ml_models)
    n_samples = [len(d[1]) for d in str_exps]


    # Get likelihoods for the different models
    ml_likelihoods = {}
    dft_likelihoods = {}
    for mol in str_dfts[0]:
        ml_likelihoods[mol] = 0
        dft_likelihoods[mol] = 0
        for i in range(n_models):
            # Get likelihoods for the model that we assume is the true one
            ml_likelihoods[mol] += ml_models[i].get_log_likelihood(str_exps[i][1], str_mls[i][mol], str_vars[i][mol])
            dft_likelihoods[mol] += dft_models[i].get_log_likelihood(str_exps[i][1], str_dfts[i][mol])

    # Draw samples
    ml_ll = {}
    dft_ll = {}
    for mol in str_dfts[0]:
        ml_ll[mol] = 0
        dft_ll[mol] = 0
        for i in range(n_models):
            for j in range(n_samples[i]):
                ml_ll[mol] += ml_models[i].get_sample_likelihoods(str_vars[i][mol][j],n=n)
                dft_ll[mol] += dft_models[i].get_sample_likelihoods(n=n)

    # Calculate probability of true model
    for mol1 in str_dfts[0]:
        llr_observed = dft_likelihoods[mol1] - np.log(sum(np.exp(ll) for ll in dft_likelihoods.values()))
        #p_sampled = np.exp(dft_ll[mol])
        p_data = 0
        for mol2 in str_dfts[0]:
            if mol1 == mol2:
                p_data += np.exp(dft_likelihoods[mol2])
                continue
            p_data += np.exp(dft_ll[mol2])
        llr_sampled = dft_likelihoods[mol1] - np.log(p_data)
        print(llr_observed, llr_sampled[:5])
        print(sum(llr_observed > llr_sampled))
        quit()



    quit()






if __name__ == "__main__":
    j_ml_model, j_dft_model, j_str_exp, j_str_dft, j_str_ml, j_str_var = J1CH_results()
    h_ml_model, h_dft_model, h_str_exp, h_str_dft, h_str_ml, h_str_var = H1_results()
    c_ml_model, c_dft_model, c_str_exp, c_str_dft, c_str_ml, c_str_var = C13_results()
    get_probability_of_true_models([j_ml_model,h_ml_model,c_ml_model],
                                   [j_dft_model,h_dft_model,c_dft_model],
                                   [j_str_exp,h_str_exp,c_str_exp],
                                   [j_str_dft,h_str_dft,c_str_dft],
                                   [j_str_ml,h_str_ml,c_str_ml],
                                   [j_str_var, h_str_var, c_str_var], n=100000)
    quit()
    print("*** mol/ll_ratio to 1 - combined data - Strychnine - ML to exp")
    for mol in j_ml_ll:
        print(mol, np.exp(j_ml_ll[1]+h_ml_ll[1]+c_ml_ll[1] - j_ml_ll[mol]-h_ml_ll[mol]-c_ml_ll[mol]))
    print("*** mol/ll_ratio to 1 - combined data - Strychnine - DFT to exp")
    for mol in j_ml_ll:
        print(mol, np.exp(j_dft_ll[1]+h_dft_ll[1]+c_dft_ll[1] - j_dft_ll[mol]-h_dft_ll[mol]-c_dft_ll[mol]))

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




