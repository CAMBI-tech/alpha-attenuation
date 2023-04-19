# Update the BciPy models with methods for calculating performance across models using sklearn's metrics module.
# This will allow us to compare models using the same metrics.
import numpy as np
# from bcipy.helpers.task import TrialReshaper
from bcipy.helpers.exceptions import SignalException
from bcipy.signal.model import ModelEvaluationReport, SignalModel
from bcipy.signal.model.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model import RdaKdeModel, PcaRdaKdeModel
from bcipy.signal.model.cross_validation import (
    cost_cross_validation_auc,
    cross_validation,
)
from bcipy.signal.model.density_estimation import KernelDensityEstimate
from bcipy.signal.model.dimensionality_reduction import (
    ChannelWisePrincipalComponentAnalysis,
    MockPCA,
)
from bcipy.signal.model.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels


class BaseRdaKdeModel(RdaKdeModel):

    def fit(self, train_data: np.array, train_labels: np.array) -> SignalModel:
        """
        Train on provided data using K-fold cross validation and return self.
        Parameters:
            train_data: shape (Channels, Trials, Trial_length) preprocessed data
            train_labels: shape (Trials,) binary labels
        Returns:
            trained likelihood model
        """
        model = Pipeline(
            [
                MockPCA(),
                RegularizedDiscriminantAnalysis(),
            ]
        )

        # Find the optimal gamma + lambda values
        arg_cv = cross_validation(train_data, train_labels, model=model, k_folds=self.k_folds)

        # Get the AUC using those optimized gamma + lambda
        rda_index = 1  # the index in the pipeline
        model.pipeline[rda_index].lam = arg_cv[0]
        model.pipeline[rda_index].gam = arg_cv[1]
        tmp, sc_cv, y_cv = cost_cross_validation_auc(
            model, rda_index, train_data, train_labels, arg_cv, k_folds=self.k_folds, split="uniform"
        )
        self.auc = -tmp
        # After finding cross validation scores do one more round to learn the
        # final RDA model
        model.fit(train_data, train_labels)

        # Insert the density estimates to the model and train using the cross validated
        # scores to avoid over fitting. Observe that these scores are not obtained using
        # the final model
        model.add(KernelDensityEstimate(scores=sc_cv))
        model.pipeline[-1].fit(sc_cv, y_cv)

        self.model = model

        if self.prior_type == "uniform":
            self.log_prior_class_1 = self.log_prior_class_0 = np.log(0.5)
        elif self.prior_type == "empirical":
            prior_class_1 = np.sum(train_labels == 1) / len(train_labels)
            self.log_prior_class_1 = np.log(prior_class_1)
            self.log_prior_class_0 = np.log(1 - prior_class_1)
        else:
            raise ValueError("prior_type must be 'empirical' or 'uniform'")

        self.classes_ = unique_labels(train_labels)
        self._ready_to_predict = True
        return self

    def evaluate(self, test_data: np.array, test_labels: np.array) -> ModelEvaluationReport:
        """Computes AUROC of the intermediate RDA step of the pipeline using k-fold cross-validation

        Args:
            test_data (np.array): shape (Channels, Trials, Trial_length) preprocessed data.
            test_labels (np.array): shape (Trials,) binary labels.

        Raises:
            SignalException: error if called before model is fit.

        Returns:
            ModelEvaluationReport: stores AUC
        """
        if not self._ready_to_predict:
            raise SignalException("must use model.fit() before model.evaluate()")

        tmp_model = Pipeline([self.model.pipeline[0], self.model.pipeline[1]])

        lam_gam = (self.model.pipeline[1].lam, self.model.pipeline[1].gam)
        tmp, _, _ = cost_cross_validation_auc(
            tmp_model, self.optimization_elements, test_data, test_labels,
            lam_gam, k_folds=self.k_folds, split="uniform"
        )
        auc = -tmp
        return ModelEvaluationReport(auc)

    def predict(self, data: np.array) -> np.array:
        """
        sklearn-compatible method for predicting
        """
        if not self._ready_to_predict:
            raise SignalException("must use model.fit() before model.predict()")

        # p(l=1 | e) = p(e | l=1) p(l=1)
        probs = self.predict_proba(data)
        return probs.argmax(-1)


class BasePcaRdaKdeModel(PcaRdaKdeModel):
    # reshaper = TrialReshaper()

    def fit(self, train_data: np.array, train_labels: np.array) -> SignalModel:
        """
        Train on provided data using K-fold cross validation and return self.
        Parameters:
            train_data: shape (Channels, Trials, Trial_length) preprocessed data
            train_labels: shape (Trials,) binary labels
        Returns:
            trained likelihood model
        """
        model = Pipeline(
            [
                ChannelWisePrincipalComponentAnalysis(n_components=self.pca_n_components, num_ch=train_data.shape[0]),
                RegularizedDiscriminantAnalysis(),
            ]
        )

        # Find the optimal gamma + lambda values
        arg_cv = cross_validation(train_data, train_labels, model=model, k_folds=self.k_folds)

        # Get the AUC using those optimized gamma + lambda
        rda_index = 1  # the index in the pipeline
        model.pipeline[rda_index].lam = arg_cv[0]
        model.pipeline[rda_index].gam = arg_cv[1]
        tmp, sc_cv, y_cv = cost_cross_validation_auc(
            model, rda_index, train_data, train_labels, arg_cv, k_folds=self.k_folds, split="uniform"
        )
        self.auc = -tmp
        # After finding cross validation scores do one more round to learn the
        # final RDA model
        model.fit(train_data, train_labels)

        # Insert the density estimates to the model and train using the cross validated
        # scores to avoid over fitting. Observe that these scores are not obtained using
        # the final model
        model.add(KernelDensityEstimate(scores=sc_cv))
        model.pipeline[-1].fit(sc_cv, y_cv)

        self.model = model

        if self.prior_type == "uniform":
            self.log_prior_class_1 = self.log_prior_class_0 = np.log(0.5)
        elif self.prior_type == "empirical":
            prior_class_1 = np.sum(train_labels == 1) / len(train_labels)
            self.log_prior_class_1 = np.log(prior_class_1)
            self.log_prior_class_0 = np.log(1 - prior_class_1)
        else:
            raise ValueError("prior_type must be 'empirical' or 'uniform'")

        self.classes_ = unique_labels(train_labels)
        self._ready_to_predict = True
        return self

    def evaluate(self, test_data: np.array, test_labels: np.array) -> ModelEvaluationReport:
        """Computes AUROC of the intermediate RDA step of the pipeline using k-fold cross-validation

        Args:
            test_data (np.array): shape (Channels, Trials, Trial_length) preprocessed data.
            test_labels (np.array): shape (Trials,) binary labels.

        Raises:
            SignalException: error if called before model is fit.

        Returns:
            ModelEvaluationReport: stores AUC
        """
        if not self._ready_to_predict:
            raise SignalException("must use model.fit() before model.evaluate()")

        tmp_model = Pipeline([self.model.pipeline[0], self.model.pipeline[1]])

        lam_gam = (self.model.pipeline[1].lam, self.model.pipeline[1].gam)
        tmp, _, _ = cost_cross_validation_auc(
            tmp_model, self.optimization_elements, test_data, test_labels,
            lam_gam, k_folds=self.k_folds, split="uniform"
        )
        auc = -tmp
        return ModelEvaluationReport(auc)

    def predict(self, data: np.array) -> np.array:
        """
        sklearn-compatible method for predicting
        """
        if not self._ready_to_predict:
            raise SignalException("must use model.fit() before model.predict()")

        # p(l=1 | e) = p(e | l=1) p(l=1)
        probs = self.predict_proba(data)
        return probs.argmax(-1)
