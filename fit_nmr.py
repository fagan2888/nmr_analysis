import numpy as np
import scipy
import scipy.stats
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt


# TODO get labels and predictions
def read_nmr(filename, col=0):
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
    unique_labels = []
    unique_predictions = []
    unique_variances = []
    for mol in sorted_mols:
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
            if i + 1 < len(labels) and label == labels[i+1]:# and 0 == 1:
                continue

            unique_labels.append(this_label / c)
            unique_predictions.append(this_prediction / c)
            unique_variances.append(this_variance / c)
            c = 0
            this_label = 0
            this_prediction = 0
            this_variance = 0

    return np.asarray(unique_labels), np.asarray(unique_predictions), np.asarray(unique_variances)

def ProbabilityModel(object):
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

    def _loss_function(self, x, errors, variances=None):
        """
        Loss function for the optimization step
        """

        n = len(errors)
        loc = x[0]
        if self._fitted_variances:
            a = np.exp(x[1])
            b = np.exp(x[2])
        else:
            scale = np.exp(x[1])
        if self.distribution == "t":
            df = np.exp(x[3 - int(self._fitted_variances)])
        else:
            df = None
        if self.scaling:
            kappa = x[-1]
        else:
            kappa = 1

        negative_log_likelihood = 0
        for i in range(n):
            if self.fitted_variances:
                scale = a * variances[i]**b
            error = kappa * errors[i]
            negative_log_likelihood -= log_probability(errors[i], scale, loc, df)

        bic = np.log(n) * len(x) + 2 * negative_log_likelihood
        return bic

    def fit(self, labels, predictions, variances=None):
        """
        Fit probability distributions to the errors, optionally utilizing individual variances
        and fitting the scale parameter to follow a power-law
        """
        errors = labels - predictions

        self._variances_fitted = True
        if variances is None:
            self._variances_fitted = False

        if self.scaling:
            slope, intercept = scipy.stats.linregress(labels, predictions)[:2]

        if self.distribution == "normal":
            loc = np.mean(errors)
            log_scale = np.log(np.std(errors))
            if not self._variances_fitted:
                if self.scaling:
                    x0 = [loc, log_scale]
                else:
                    self.params = (loc, log_scale)
                    return
            x0 = [loc, log_scale, 0]
        elif self.distribution in ["cauchy", "t"]:
            sorted_errors = np.sort(errors)
            n = sorted_errors.size
            # 24% mid to do estimates
            start = int(0.38 * n+ 1 )
            end = int(0.62 * n + 1)
            loc = np.mean(sorted_errors[start:mean])
            # half the interquartile range as estimator of scale
            log_scale = np.log((np.percentile(sorted_errors, 75) - np.percentile(sorted_errors,25)) / 2)
            if self._variances_fitted:
                x0 = [loc, log_scale, 0]
            else:
                x0 = [loc, log_scale]
            if self.distribution == "t":
                x0.append(2)
        elif self.distribution == "laplace":
            loc = np.median(errors)
            log_scale = np.log(np.mean(abs(errors - loc)))
            if not self._variances_fitted:
                if self.scaling:
                    x0 = [loc, log_scale]
                else:
                    self.params = (loc, log_scale)
                    return
            x0 = [loc, log_scale, 0]

        if self.scaling:
            x0.append(slope)
            print(x0[0], intercept)
            quit()

        res = scipy.optimize.minimize(self._loss_function, x0, args=(errors, variances), method="bfgs")
        self.params = res.x
        self.bic = res.fun
        return self

    def get_log_likelihood(errors, variances=None):
        """
        Get the log-likelihood of some prediction errors and optionally variances
        """
        #TODO
        quit()


if __name__ == "__main__":
    labels, predictions, variances = read_nmr("__JUL03___SYG_exp_1JCH_raw.txt")
    quit()
    fit(coupling_errors, coupling_variances, form="laplace")
    fit2(coupling_errors, coupling_variances, form="laplace")
    fitt(coupling_errors, coupling_variances)
    shift_errors, shift_variances = read_nmr("__JUL03___CD_exp_CCS_raw.txt", -1)
    fit(shift_errors, shift_variances, form="laplace")
    fit2(shift_errors, shift_variances, form="laplace")
    fitt(shift_errors, shift_variances)
