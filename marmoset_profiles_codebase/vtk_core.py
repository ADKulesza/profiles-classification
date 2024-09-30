#!/usr/bin/env python
# encoding: utf-8


__author__ = "Piotr Majka"
__maintainer__ = "Piotr Majka"
__email__ = "pmajka@nencki.gov.pl"

import numpy as np
import vtk


def points_to_vtk_points(points_list):
    """
    TODO: Doc, Tests
    """
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()

    for i in range(len(points_list)):
        id_ = points.InsertNextPoint(points_list[i])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id_)

    point = vtk.vtkPolyData()
    point.SetPoints(points)
    point.SetVerts(vertices)

    return point


def load_vtk_image(filename):
    image_reader = vtk.vtkStructuredPointsReader()
    image_reader.SetFileName(filename)
    image_reader.Update()
    return image_reader


def load_vtk_polydata(filename):
    poly_data_reader = vtk.vtkPolyDataReader()
    poly_data_reader.SetFileName(filename)
    poly_data_reader.Update()

    return poly_data_reader.GetOutput()


def save_vtk_polydata(filename, polydata):
    poly_data_writer = vtk.vtkPolyDataWriter()
    try:
        poly_data_writer.SetInput(polydata)
    except:
        poly_data_writer.SetInputData(polydata)
    poly_data_writer.SetFileName(filename)
    poly_data_writer.Update()


def get_line(x, teval):
    """
    TODO: Document this method end expend it so it is better docymmented
    and a bit more flexible in terms which data is supplied to this function.
    """
    # TODO: The method should be able to determine the actual number of
    # points is the given dataset, don't you thing so?
    n = int(teval)

    points = vtk.vtkPoints()
    if type(points) == type(x):
        points = x
    else:
        for i in range(n):
            points.InsertNextPoint(*x[i])

    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(n)
    for i in range(n):
        line.GetPointIds().SetId(i, i)

    cell_array = vtk.vtkCellArray()
    cell_array.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(cell_array)

    return polydata


def get_vtk_poly_line_length(polydata, start_index=None, end_index=None):
    """
    Calculate the length of a line provided as `vtk.vtkLine` object.
    It is not asserted in the function of the provided cell is actually
    a `vtk.vtkLine` object. it is up to the user to provide correct
    data structure.

    :param polydata: A vtk polygonal data comprising a single cell which
        is of a `vtk.vtkLine` object.
    :type polydata: `vtk.vtkPolyData`

    :param start_index: The index of a point of the polyline starting
        from which the distance will be calculated. This index has to be
        lower than the `end_index` and, obviously not smaller than zero.
    :type start_index: int

    :param end_index: The index of the point of the polyline on which
        calculating the length on the poly line would end. This is to
        be used if you want to calculate the distance of only part
        of the polyline. This value has to be larger than one and
        larger than the value of the `start_index`.
    :type end_index: int

    :return:
    :rtype:
    """
    # TODO: This has to be put in some auxiliary module.

    assert (
        start_index > 0 and start_index < end_index
    ), "Incorrect value of the start_index argument."

    assert (
        end_index > 1 and end_index > start_index
    ), "Incorrect value of the end_index argument."

    distance_fnc = vtk.vtkMath.Distance2BetweenPoints
    number_of_points = polydata.GetNumberOfPoints()

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = number_of_points - 1

    total_length = 0
    last_point = polydata.GetPoint(0)

    for point_id in range(number_of_points):
        point_location = polydata.GetPoint(point_id)
        total_length += np.sqrt(distance_fnc(point_location, last_point))
        last_point = point_location

    return total_length
