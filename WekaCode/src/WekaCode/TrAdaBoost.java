package WekaCode;

import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class TrAdaBoost extends IteratedSingleClassifierEnhancer {

        private static final long serialVersionUID = 6153382457807171663L;
        private Instances m_target = null;
        private double[] m_betas;
        private int m_NumIterationsPerformed;

        public TrAdaBoost(Instances target) {
                super();
                m_target = new Instances(target);
        }

        @Override
        public void buildClassifier(Instances data) throws Exception {
                super.buildClassifier(data);
                performBuild(data, m_target);
        }

        private void performBuild(Instances source, Instances target)
                        throws Exception {

                if (target == null)
                        throw new Exception("not target to work on");

                m_betas = new double[m_NumIterations];
                m_NumIterationsPerformed = 0;
                int sourceInstances = source.numInstances();
                int numInstances = sourceInstances + target.numInstances();
                double[][] weights = new double[m_NumIterations + 1][numInstances];
                for (int i = 0; i < weights[0].length; ++i)
                        weights[0][i] = 1;
                double beta = 1 / (1 + Math.sqrt(2 * Math.log(sourceInstances
                                / (0.0 + m_NumIterations))));

                Instances data = join(source, target);
                for (int iteration = 0; iteration < m_NumIterations; ++iteration) {

                        // Set weights and build current iteration classifier
                        weighData(data, weights[iteration]);
                        Instances trainData = new Instances(data);
                        m_Classifiers[iteration].buildClassifier(trainData);
                        trainData = null;
                        Classifier classifier = m_Classifiers[iteration];

                        // Find epsilon_t and beta_t
                        double targetWeights = 0;
                        double epsilon = 0;
                        for (int i = sourceInstances; i < numInstances; ++i) {
                                Instance instance = data.instance(i);
                                double weight = instance.weight();
                                targetWeights += weight;
                                if (!Utils.eq(classifier.classifyInstance(instance),
                                                instance.classValue()))
                                        epsilon += weight;
                        }
                        epsilon /= targetWeights;
                        m_betas[iteration] = epsilon / (1 - epsilon);

                        // Stop if error too big or 0
                        if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0)) {
                                if (iteration == 0) {
                                        // If we're the first we have to use it
                                        m_NumIterationsPerformed = 1;
                                        m_betas[iteration] = 1;
                                }
                                break;
                        } else {
                                m_NumIterationsPerformed = iteration + 1;
                        }

                        // Set next iteration weights
                        for (int i = 0; i < sourceInstances; ++i) {
                                Instance instance = data.instance(i);
                                weights[iteration + 1][i] = weights[iteration][i];
                                if (!Utils.eq(classifier.classifyInstance(instance),
                                                instance.classValue()))
                                        weights[iteration + 1][i] *= beta;
                        }
                        for (int i = sourceInstances; i < numInstances; ++i) {
                                Instance instance = data.instance(i);
                                weights[iteration + 1][i] = weights[iteration][i];
                                if (!Utils.eq(classifier.classifyInstance(instance),
                                                instance.classValue()))
                                        weights[iteration + 1][i] /= m_betas[iteration];
                        }
                }

        }

        private void weighData(Instances data, double[] weights) {

                if (weights.length != data.numInstances()) {
                        System.err.println("data and weights are different length");
                        System.exit(1);
                }

                for (int i = 0; i < weights.length; ++i)
                        data.instance(i).setWeight(weights[i]);

        }

        @SuppressWarnings("unchecked")
        private Instances join(Instances source, Instances target) {

                int numInstances = source.numInstances() + target.numInstances();
                Instances retVal = new Instances(source, numInstances);

                Enumeration<Instance> enu = source.enumerateInstances();
                while (enu.hasMoreElements())
                        retVal.add(enu.nextElement());

                enu = target.enumerateInstances();
                while (enu.hasMoreElements())
                        retVal.add(enu.nextElement());

                return retVal;
        }

        @Override
        public double classifyInstance(Instance instance) throws Exception {

                if (m_NumIterationsPerformed <= 0)
                        throw new Exception("No model built");

                double sumClassified = 0;
                double sumBase = 0;

                for (int t = (m_NumIterationsPerformed - 1) / 2; t < m_NumIterationsPerformed; ++t) {
                        double beta = m_betas[t];
                        Classifier classifier = m_Classifiers[t];
                        double classification = classifier.classifyInstance(instance);
                        if (!Utils.eq(classification, 0.0))
                                sumClassified -= Math.log(beta);
                        sumBase -= Math.log(Math.sqrt(beta));
                }

                if (sumClassified >= sumBase)
                        return 1.0;
                return 0.0;
        }

        @Override
        public String toString() {

                StringBuilder text = new StringBuilder();

                if (m_NumIterationsPerformed <= 0) {
                        text.append("TrAdaBoost: No model built yet.\n");
                } else if (m_NumIterationsPerformed == 1) {
                        text.append("TrAdaBoost: No boosting possible, one classifier used!\n");
                        text.append(m_Classifiers[0].toString() + "\n");
                } else {
                        text.append("TrAdaBoost: Base classifiers and their weights: \n\n");
                        for (int i = 0; i < m_NumIterationsPerformed; i++) {
                                text.append(m_Classifiers[i].toString() + "\n\n");
                                text.append("Weight: " + Utils.roundDouble(m_betas[i], 2)
                                                + "\n\n");
                        }
                        text.append("Number of performed Iterations: "
                                        + m_NumIterationsPerformed + "\n");
                }
                return text.toString();
        }
}
