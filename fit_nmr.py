import numpy as np
import scipy
import scipy.stats
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt
import seaborn as sns

def read_nmr(filename, col=0, concatenate=True, ignore_similar=True):
    # Parse
    with open(filename) as f:
        lines = f.readlines()

    data = {}
    mols = []
    labels = []
    predictions = []
    variances = []
    for line in lines[1:]:
        tokens = line.split()
        mol = int(tokens[0])
        label = float(tokens[4+col])
        prediction = float(tokens[5+col])
        variance = float(tokens[6+col])
        if mol not in data:
            data[mol] = {}
            data[mol]["labels"] = []
            data[mol]["predictions"] = []
            data[mol]["variances"] = []
        data[mol]["labels"].append(label)
        data[mol]["predictions"].append(prediction)
        data[mol]["variances"].append(variance)

    sorted_mols = sorted(data.keys())

    # Average duplicates
    unique_labels = {}
    unique_predictions = {}
    unique_variances = {}
    for mol in sorted_mols:
        unique_labels[mol] = []
        unique_predictions[mol] = []
        unique_variances[mol] = []
        sorted_idx = np.argsort(data[mol]["labels"])
        labels = np.asarray(data[mol]["labels"])[sorted_idx]
        predictions = np.asarray(data[mol]["predictions"])[sorted_idx]
        variances = np.asarray(data[mol]["variances"])[sorted_idx]
        c = 0
        this_label = 0
        this_prediction = 0
        this_variance = 0
        for i, (label, prediction, variance) in enumerate(zip(labels, predictions, variances)):
            this_label += label
            this_prediction += prediction
            this_variance += variance
            c += 1
            if ignore_similar and i + 1 < len(labels) and label == labels[i+1]:# and 0 == 1:
                continue

            unique_labels[mol].append(this_label / c)
            unique_predictions[mol].append(this_prediction / c)
            unique_variances[mol].append(this_variance / c)
            c = 0
            this_label = 0
            this_prediction = 0
            this_variance = 0


    if concatenate:
        return np.asarray(sum(unique_labels.values(), [])), \
                np.asarray(sum(unique_predictions.values(), [])), \
                np.asarray(sum(unique_variances.values(), []))

    # Convert to numpy arrays
    for mol in unique_labels.keys():
        unique_labels[mol] = np.asarray(unique_labels[mol])
        unique_predictions[mol] = np.asarray(unique_predictions[mol])
        unique_variances[mol] = np.asarray(unique_variances[mol])

    return unique_labels, unique_predictions, unique_variances

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
        if variances is None and self._variances_fitted is not None:
            raise SystemExit("Variances has to be fitted to be able to used to get log-likelihood")
        elif variances is not None and self._variances_fitted is None:
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
                return sum(self._log_probability(error, scale=scale, df=df, loc=loc) for (error, variance) \
                        in zip(errors, variances))
        else:
            if self._variances_fitted:
                loc, log_a, b = self.params[:end]
                a = np.exp(log_a)
                return sum(self._log_probability(error, scale=a*variance**b, loc=loc) for (error, variance) \
                        in zip(errors, variances))
            else:
                loc, log_scale = self.params[:end]
                scale = np.exp(log_scale)
                return sum(self._log_probability(error, scale=scale, loc=loc) for (error, variance) \
                        in zip(errors, variances))

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

def J1CH_results():
    syg_exp, syg_ml, variances = read_nmr("__JUL07___SYG_exp_1JCH_raw.txt", ignore_similar=False)
    syg_dft = read_nmr("__JUL07___SYG_dft_1JCH_raw.txt", ignore_similar=False)[0]

    errors = syg_exp - 10.91 - syg_ml
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_ml)[variances < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_ml)[variances < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - ML to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    errors = syg_exp - 10.91 - syg_dft
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 10.91 Hz offset - no variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_dft)[variances < 10]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))
    errors = (syg_exp - 10.91 - syg_dft)[variances < 5]
    print("*** MAE/RMSE/MaxE - Sygenta - DFT to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    print(mae(errors), rmse(errors), maxe(errors))

    str_dft, str_ml, str_variances = read_nmr("__JUL07___STRYCH_dft_1JCH_raw.txt", concatenate=False)
    str_exp = read_nmr("STR_J1CH_val_raw.txt", -1, concatenate=False)[0]
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 10.91 Hz offset - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - 10.91 - str_ml[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 10.91 - str_ml[mol])[str_variances < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - ML to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 10.91 - str_ml[mol])[str_variances < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))

    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 10.91 Hz offset - no variance filter ***")
    for mol in str_dft.keys():
        errors = str_exp[1] - 10.91 - str_dft[mol]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 10.91 Hz offset - 10 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 10.91 - str_dft[mol])[str_variances < 10]
        print(mol, mae(errors), rmse(errors), maxe(errors))
    print("*** MOL/MAE/RMSE/MaxE - Strychnine - DFT to exp - 10.91 Hz offset - 5 Hz variance filter ***")
    for mol in str_dft.keys():
        errors = (str_exp[1] - 10.91 - str_dft[mol])[str_variances < 5]
        print(mol, mae(errors), rmse(errors), maxe(errors))

    syg_exp, syg_ml, variances = read_nmr("__JUL07___SYG_exp_1JCH_raw.txt")
    syg_dft = read_nmr("__JUL07___SYG_dft_1JCH_raw.txt")[0]
    ml_model = ProbabilityModel(distribution='laplace', scaling=False)
    ml_model.fit(syg_exp, syg_ml, variances)
    dft_model = ProbabilityModel(distribution='laplace', scaling=False)
    dft_model.fit(syg_exp, syg_dft)

    # Print all the things
    print("*** Parameters - Sygenta - ML to exp ***")
    print(ml_model.params)
    print("*** Parameters - Sygenta - DFT to exp ***")
    print(dft_model.params)


    ml_ll = []
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - ML to exp ***")
    ll1 = ml_model.get_log_likelihood(str_exp[1], str_ml[1], str_variances[1])
    for mol in str_dft.keys():
        ll = ml_model.get_log_likelihood(str_exp[1], str_ml[mol], str_variances[mol])
        ml_ll.append(ll)
        print(mol, ll, np.exp(ll1-ll))

    dft_ll = []
    print("*** mol/log-likelihood/ratio to 1 - Strychnine - DFT to exp ***")
    ll1 = ml_model.get_log_likelihood(str_exp[1], str_dft[1])
    for mol in str_dft.keys():
        ll = ml_model.get_log_likelihood(str_exp[1], str_dft[mol])
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




