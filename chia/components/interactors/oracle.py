from chia.components.interactors import interactor


class OracleInteractor(interactor.Interactor):
    def query_annotations_for(self, samples, gt_resource_id, ann_resource_id):
        return [
            sample.add_resource(
                self.__class__.__name__,
                ann_resource_id,
                sample.get_resource(gt_resource_id),
            )
            for sample in samples
        ]
