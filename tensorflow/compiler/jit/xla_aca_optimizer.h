#ifndef TENSORFLOW_COMPILER_JIT_XLA_ACA_OPTIMIZER_H_
#define TENSORFLOW_COMPILER_JIT_XLA_ACA_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

namespace tensorflow {

// We are optimizing graphs by fusing operations when a certain sequence is found, resulthe result is a more 
// efficient execution.
class XlaACAOptimizer : public grappler::CustomGraphOptimizer {
 public:
  XlaACAOptimizer() {}
  ~XlaACAOptimizer() override {}

  Status Init(
      const RewriterConfig_CustomGraphOptimizer* config = nullptr) override {
    return Status::OK();
  }

  string name() const override { return "xla-fusion"; };

  Status Optimize(grappler::Cluster* cluster,
                  const grappler::GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(grappler::Cluster* cluster, const grappler::GrapplerItem& item,
                const GraphDef& optimize_output, double result) override {
  }
};

}  // namespace tensorflow

#endif // TENSORFLOW_COMPILER_JIT_XLA_ACA_OPTIMIZER_H_