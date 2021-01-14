/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <atomic>
#include <memory>
#include <mutex>


#include <ginkgo/core/base/machine_topology.hpp>


namespace gko {


namespace detail {


std::shared_ptr<const MachineTopology> machine_topology{};
std::mutex machine_topology_mutex{};
std::atomic<bool> initialized_machine_topology{};


}  // namespace detail


const MachineTopology *get_machine_topology()
{
    if (!detail::initialized_machine_topology.load()) {
        std::lock_guard<std::mutex> guard(detail::machine_topology_mutex);
        if (!detail::machine_topology) {
            detail::machine_topology = MachineTopology::create();
            detail::initialized_machine_topology.store(true);
        }
    }
    assert(detail::machine_topology.get() != nullptr);
    return detail::machine_topology.get();
}


const MachineTopology::io_obj_info *MachineTopology::get_pci_device(
    const std::string &pci_bus_id) const
{
    for (auto id = 0; id < this->pci_devices_.size(); ++id) {
        if (this->pci_devices_[id].pci_bus_id.compare(0, 12, pci_bus_id, 0,
                                                      12) == 0) {
            return &this->pci_devices_[id];
        }
    }
    return nullptr;
}


MachineTopology::MachineTopology()
{
#if GKO_HAVE_HWLOC

    // Initialize the topology from hwloc
    this->topo_ =
        hwloc_manager<hwloc_topology>(init_topology(), hwloc_topology_destroy);
    // load objects of type Package . See HWLOC_OBJ_PACKAGE for more details.
    load_objects(HWLOC_OBJ_PACKAGE, this->packages_);
    // load objects of type NUMA Node. See HWLOC_OBJ_NUMANODE for more details.
    load_objects(HWLOC_OBJ_NUMANODE, this->numa_nodes_);
    // load objects of type Core. See HWLOC_OBJ_CORE for more details.
    load_objects(HWLOC_OBJ_CORE, this->cores_);
    // load objects of type processing unit(PU). See HWLOC_OBJ_PU for more
    // details.
    load_objects(HWLOC_OBJ_PU, this->pus_);
    // load objects of type PCI Devices See HWLOC_OBJ_PCI_DEVICE for more
    // details.
    load_objects(HWLOC_OBJ_PCI_DEVICE, this->pci_devices_);
    num_numas_ = hwloc_get_nbobjs_by_type(this->topo_.get(), HWLOC_OBJ_PACKAGE);

#else

    this->topo_ = hwloc_manager<hwloc_topology>();

#endif
}


void MachineTopology::hwloc_binding_helper(
    const std::vector<MachineTopology::normal_obj_info> &obj, const int *id,
    const size_type num_ids) const
{
#if GKO_HAVE_HWLOC
    auto bitmap_toset = hwloc_bitmap_alloc();
    // Set the given ids to a bitmap
    for (auto i = 0; i < num_ids; ++i) {
        GKO_ASSERT(id[i] < obj.size());
        GKO_ASSERT(id[i] >= 0);
        hwloc_bitmap_set(bitmap_toset, obj[id[i]].os_id);
    }

    // Singlify to reduce expensive migrations.
    hwloc_bitmap_singlify(bitmap_toset);
    hwloc_set_cpubind(this->topo_.get(), bitmap_toset, 0);
    hwloc_bitmap_free(bitmap_toset);
#endif
}


void MachineTopology::load_objects(
    hwloc_obj_type_t type,
    std::vector<MachineTopology::normal_obj_info> &objects) const
{
#if GKO_HAVE_HWLOC
    // Get the number of normal objects of a certain type (Core, PU, Machine
    // etc.).
    unsigned num_objects = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    for (unsigned i = 0; i < num_objects; i++) {
        // Get the actual normal object of the given type.
        hwloc_obj_t obj = hwloc_get_obj_by_type(this->topo_.get(), type, i);
        objects.push_back(normal_obj_info{obj, obj->logical_index,
                                          obj->os_index, obj->gp_index,
                                          hwloc_bitmap_first(obj->nodeset)});
    }
#endif
}


inline int MachineTopology::get_obj_id_by_os_index(
    const std::vector<MachineTopology::normal_obj_info> &objects,
    size_type os_index) const
{
#if GKO_HAVE_HWLOC
    for (auto id = 0; id < objects.size(); ++id) {
        if (objects[id].os_id == os_index) {
            return id;
        }
    }
#endif
    return -1;
}


inline int MachineTopology::get_obj_id_by_gp_index(
    const std::vector<MachineTopology::normal_obj_info> &objects,
    size_type gp_index) const
{
#if GKO_HAVE_HWLOC
    for (auto id = 0; id < objects.size(); ++id) {
        if (objects[id].gp_id == gp_index) {
            return id;
        }
    }
#endif
    return -1;
}


void MachineTopology::load_objects(
    hwloc_obj_type_t type,
    std::vector<MachineTopology::io_obj_info> &vector) const
{
#if GKO_HAVE_HWLOC
    GKO_ASSERT(this->cores_.size() != 0);
    GKO_ASSERT(this->pus_.size() != 0);
    unsigned num_objects = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    for (unsigned i = 0; i < num_objects; i++) {
        // Get the actual PCI object.
        hwloc_obj_t obj = hwloc_get_obj_by_type(this->topo_.get(), type, i);
        // Get the non-IO ancestor (which is the closest and the one that can be
        // bound to) of the object.
        auto ancestor = hwloc_get_non_io_ancestor_obj(this->topo_.get(), obj);
        // Create the object.
        vector.push_back(
            io_obj_info{obj, obj->logical_index, obj->os_index, obj->gp_index,
                        hwloc_bitmap_first(ancestor->nodeset), ancestor});
        // Get the corresponding cpuset of the ancestor nodeset
        hwloc_cpuset_t ancestor_cpuset = hwloc_bitmap_alloc();
        hwloc_cpuset_from_nodeset(this->topo_.get(), ancestor_cpuset,
                                  ancestor->nodeset);
        // Find the cpu objects closest to this device from the ancestor cpuset
        // and store their ids for binding purposes
        int closest_cpu_id = -1;
        int closest_os_id = hwloc_bitmap_first(ancestor_cpuset);
        // clang-format off
        hwloc_bitmap_foreach_begin(closest_os_id, ancestor_cpuset)
            closest_cpu_id = get_obj_id_by_os_index(this->pus_, closest_os_id);
            vector.back().closest_cpu_ids.push_back(closest_cpu_id);
        hwloc_bitmap_foreach_end();
        // clang-format on

        // Get local id of the ancestor object.
        if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_PACKAGE) == 0) {
            vector.back().ancestor_local_id =
                get_obj_id_by_gp_index(this->packages_, ancestor->gp_index);
        } else if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_CORE) == 0) {
            vector.back().ancestor_local_id =
                get_obj_id_by_gp_index(this->cores_, ancestor->gp_index);
        } else if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_NUMANODE) ==
                   0) {
            vector.back().ancestor_local_id =
                get_obj_id_by_gp_index(this->numa_nodes_, ancestor->gp_index);
        }
        hwloc_bitmap_free(ancestor_cpuset);
        // Get type of the ancestor object and store it as a string.
        char ances_type[24];
        hwloc_obj_type_snprintf(ances_type, sizeof(ances_type), ancestor, 0);
        vector.back().ancestor_type = std::string(ances_type);
        // Write the PCI Bus ID from the object info.
        char pci_bus_id[13];
        snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.%01x",
                 obj->attr->pcidev.domain, obj->attr->pcidev.bus,
                 obj->attr->pcidev.dev, obj->attr->pcidev.func);
        vector.back().pci_bus_id = std::string(pci_bus_id);
    }
#endif
}


hwloc_topology *MachineTopology::init_topology()
{
#if GKO_HAVE_HWLOC
    hwloc_topology_t tmp;
    hwloc_topology_init(&tmp);

    hwloc_topology_set_io_types_filter(tmp, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_set_type_filter(tmp, HWLOC_OBJ_BRIDGE,
                                   HWLOC_TYPE_FILTER_KEEP_NONE);
    hwloc_topology_set_type_filter(tmp, HWLOC_OBJ_OS_DEVICE,
                                   HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_set_xml(tmp, GKO_HWLOC_XMLFILE);
    hwloc_topology_load(tmp);

    return tmp;
#else
    // MSVC complains if there is no return statement.
    return nullptr;
#endif
}


}  // namespace gko
