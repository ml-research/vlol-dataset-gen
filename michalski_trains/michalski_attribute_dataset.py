import torch

from michalski_trains.michalski_dataset import MichalskiDataset


class MichalskiAttributeDataset(MichalskiDataset):
    ''' Dataset for the Michalski train dataset with attributes as labels.
        parameters:
        fixed_output_car_size: if set to None, the output tensor will have the size of the number of cars * attributes.
         Else the output tensor y will have the size of fixed_output_car_size * number of attributes
    '''

    def __int__(self, fixed_car_size=4, **kwargs):
        self.fixed_car_size = fixed_car_size
        super().__init__(**kwargs)

    def __getitem__(self, item):
        image = self.get_pil_image(item)
        X = self.norm(image)
        # y = self.get_attributes(item) if self.fixed_output_car_size is None else self.get_attributes_fixed_size(item)
        y = self.get_attributes_fixed_size(item)
        return X, y

    def get_attributes(self, item):
        ''' Returns a tensor of labels for all attributes of each car in the train
        :param item: index of the train
        :return: tensor of labels
        '''
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = []
        # each train has (n cars a 8 attributes) totalling to n*8 labels
        # each label can have 22 classes
        for car in cars:
            # index 0 = not existent
            color = att.index(car.get_blender_car_color())
            length = att.index(car.get_car_length())
            wall = att.index(car.get_blender_wall())
            roof = att.index(car.get_blender_roof())
            wheels = att.index(car.get_car_wheels())
            l_shape = att.index(car.get_blender_payload())
            l_num = car.get_load_number()
            l_shapes = [l_shape] * l_num + [0] * (3 - l_num)
            car_number = car.get_car_number()
            labels += [color, length, wall, roof, wheels] + l_shapes
        return torch.tensor(labels, dtype=torch.int64)

    def get_attributes_fixed_size(self, item):
        ''' Returns a tensor of labels for all attributes of each car in the train
        :param item: index of the train
        :return: tensor of labels
        '''
        # fixed_car_size = self.fixed_car_size
        fixed_car_size = 4
        # self.fixed_output_car_size = 4
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = []
        # each train has (fixed_output_car_size cars a 8 attributes) totalling to fixed_output_car_size*8 labels
        # each label can have 22 classes
        for car in cars:
            # index 0 = not existent
            color = att.index(car.get_blender_car_color())
            length = att.index(car.get_car_length())
            wall = att.index(car.get_blender_wall())
            roof = att.index(car.get_blender_roof())
            wheels = att.index(car.get_car_wheels())
            l_shape = att.index(car.get_blender_payload())
            l_num = car.get_load_number()
            l_shapes = [l_shape] * l_num + [0] * (3 - l_num)
            car_number = car.get_car_number()
            labels += [color, length, wall, roof, wheels] + l_shapes
        if len(labels) > fixed_car_size * 8:
            raise ValueError(f'Number of labels is greater than the fixed output car size.')
        labels = labels + [0] * (fixed_car_size * 8 - len(labels))
        return torch.tensor(labels, dtype=torch.int64)

    def get_ds_labels(self):
        return self.attributes

    def get_ds_classes(self):
        return self.attribute_classes

    def get_class_dim(self):
        return len(self.attribute_classes)

    def get_output_dim(self):
        return len(self.attributes)
