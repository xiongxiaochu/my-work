#!/usr/bin/python
# -*- coding: utf-8 -*-


def read_gowalla_data(file_path):
    train_file = open(file_path, 'r')
    x_data = {}
    for i in open(file_path):
        line = train_file.readline()
        # line = line.strip('\n')
        if len(line) == 0:
            continue

        items = line.split("\t")
        user_id = items[0]
        if len(user_id) == 0:
            continue
        user_id = int(user_id)
        if user_id not in x_data.keys():
            x_data[user_id] = list()
        poi_id = items[4]
        if len(poi_id) == 0:
            continue
        else:
            poi_id = int(poi_id)
            x_data[user_id].append(poi_id)
    train_file.close()
    return x_data


def read_foursquare_data(file_path):
    train_file = open(file_path, 'r')
    x_data = {}
    while 1:
        line = train_file.readline()
        if not line:
            break

        if len(line) == 0:
            continue

        items = line.split("\t")
        user_id = items[0].split("_")[1]
        if len(user_id) == 0:
            continue
        user_id = int(user_id)
        if user_id not in x_data.keys():
            x_data[user_id] = list()
        poi_id = items[1].split("_")[1]
        if len(poi_id) == 0:
            continue
        else:
            poi_id = int(poi_id)
            x_data[user_id].append(poi_id)
    train_file.close()

    return x_data


def read_gtd_data(file_path):
    train_file = open(file_path, 'r')
    x_data = {}
    while 1:
        line = train_file.readline()
        if not line:
            break

        if len(line) == 0:
            continue

        items = line.split("\t")
        user_id = items[6]
        if len(user_id) == 0:
            continue
        user_id = int(user_id)
        if user_id not in x_data.keys():
            x_data[user_id] = list()
        poi_id = items[2]
        if len(poi_id) == 0:
            continue
        else:
            poi_id = int(poi_id)
            x_data[user_id].append(poi_id)
    train_file.close()

    return x_data


def read_foursquare_users():
    users = set()
    t_file = open('../foursquare/foursquare_records.txt', 'r')
    for i in open('../foursquare/foursquare_records.txt'):
        line = t_file.readline()
        items = line.split("\t")
        user_id = int(items[0].split("_")[1])
        users.add(user_id)
    t_file.close()
    num_users = len(users)
    return num_users


def read_foursquare_pois():
    pois = set()
    t_file = open('../foursquare/foursquare_records.txt', 'r')
    for i in open('../foursquare/foursquare_records.txt'):
        line = t_file.readline()
        items = line.split("\t")
        poi_id = int(items[1].split("_")[1])
        pois.add(poi_id)
    t_file.close()
    num_pois = len(pois)
    return num_pois


def read_gtd_users():
    users = set()
    t_file = open('../GTD/old_GTD-1335/indexed_GTD.txt', 'r')
    for i in open('../GTD/old_GTD-1335/indexed_GTD.txt'):
        line = t_file.readline()
        items = line.split("\t")
        user_id = int(items[6])
        users.add(user_id)
    t_file.close()
    num_users = len(users)
    return num_users


def read_gtd_pois():
    pois = set()
    t_file = open('../GTD/old_GTD-1335/indexed_GTD.txt', 'r')
    for i in open('../GTD/old_GTD-1335/indexed_GTD.txt'):
        line = t_file.readline()
        items = line.split("\t")
        poi_id = int(items[2])
        pois.add(poi_id)
    t_file.close()
    num_pois = len(pois)
    return num_pois


def read_gowalla_users():
    users = set()
    t_file = open('../gowalla/sorted_indexed_final_gowalla.txt', 'r')
    for i in open('../gowalla/sorted_indexed_final_gowalla.txt'):
        line = t_file.readline()
        items = line.split("\t")
        user_id = int(items[0])
        users.add(user_id)
    t_file.close()
    num_users = len(users)
    return num_users


def read_gowalla_pois():
    pois = set()
    t_file = open('../gowalla/sorted_indexed_final_gowalla.txt', 'r')
    for i in open('../gowalla/sorted_indexed_final_gowalla.txt'):
        line = t_file.readline()
        items = line.split("\t")
        poi_id = int(items[4])
        pois.add(poi_id)
    t_file.close()
    num_pois = len(pois)
    return num_pois
