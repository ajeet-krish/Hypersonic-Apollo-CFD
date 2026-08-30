"""VTU file parser for SU2 hypersonic solution data.

Handles both ASCII and binary (appended) VTU formats from SU2 v8.x.
Extracts flow field data (Mach, Pressure, Temperature, Density) and
provides stagnation-point extraction and surface heat flux utilities.
"""
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VTUData:
    """Parsed VTU file data.

    Attributes:
        coordinates: Node coordinates, shape (N, 3).
        point_data: Dictionary of point data arrays, keyed by field name.
        cell_data: Dictionary of cell data arrays, keyed by field name.
    """

    coordinates: np.ndarray
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def mach(self) -> np.ndarray | None:
        """Mach number array, or None if not present."""
        return self.point_data.get("Mach")

    @property
    def pressure(self) -> np.ndarray | None:
        """Static pressure array, or None if not present."""
        return self.point_data.get("Pressure")

    @property
    def temperature(self) -> np.ndarray | None:
        """Static temperature array, or None if not present."""
        return self.point_data.get("Temperature")

    @property
    def density(self) -> np.ndarray | None:
        """Density array, or None if not present."""
        return self.point_data.get("Density")


def parse_vtu(vtu_path: Path) -> VTUData:
    """Parse SU2 VTU solution file.

    Handles both ASCII and binary (appended) VTU formats.

    Args:
        vtu_path: Path to VTU file.

    Returns:
        VTUData with extracted fields.

    Raises:
        ValueError: If the VTU file cannot be parsed.
    """
    with open(vtu_path, "rb") as f:
        content = f.read()

    if b"<AppendedData" in content:
        return _parse_vtu_appended(vtu_path)
    return _parse_vtu_ascii(vtu_path)


def _parse_vtu_ascii(vtu_path: Path) -> VTUData:
    """Parse ASCII VTU file."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(vtu_path)
    root = tree.getroot()

    coords_list: list[np.ndarray] = []
    point_fields: dict[str, np.ndarray] = {}
    cell_fields: dict[str, np.ndarray] = {}

    for piece in root.iter("Piece"):
        # Get coordinates
        points_elem = piece.find("Points")
        if points_elem is not None:
            coords_array = points_elem.find("DataArray")
            if coords_array is not None and coords_array.text:
                coords = np.fromstring(coords_array.text, sep=" ").reshape(-1, 3)
                coords_list.append(coords)

        # Get point data fields
        point_data = piece.find("PointData")
        if point_data is not None:
            for data_array in point_data.findall("DataArray"):
                name = data_array.get("Name")
                if name and data_array.text:
                    raw = np.fromstring(data_array.text, sep=" ")
                    n_components = data_array.get("NumberOfComponents")
                    if n_components is not None:
                        nc = int(n_components)
                        if len(raw) % nc == 0:
                            raw = raw.reshape(-1, nc)
                    point_fields[name] = raw

        # Get cell data fields
        cell_data = piece.find("CellData")
        if cell_data is not None:
            for data_array in cell_data.findall("DataArray"):
                name = data_array.get("Name")
                if name and data_array.text:
                    cell_fields[name] = np.fromstring(data_array.text, sep=" ")

    if not coords_list:
        raise ValueError(f"No coordinates found in {vtu_path}")

    return VTUData(
        coordinates=np.vstack(coords_list),
        point_data=point_fields,
        cell_data=cell_fields,
    )


def _parse_vtu_appended(vtu_path: Path) -> VTUData:
    """Parse VTU file with appended binary data."""
    import xml.etree.ElementTree as ET

    with open(vtu_path, "rb") as f:
        binary_content = f.read()

    # Find end of XML
    xml_end_marker = b"</UnstructuredGrid>"
    xml_end_pos = binary_content.find(xml_end_marker)
    if xml_end_pos == -1:
        raise ValueError(f"No closing UnstructuredGrid tag in {vtu_path}")

    xml_end_pos += len(xml_end_marker)

    # Parse XML header
    xml_header = binary_content[:xml_end_pos].decode("utf-8")
    root = ET.fromstring(xml_header + "</VTKFile>")

    # Find start of binary data (after "_" in AppendedData)
    appended_tag_start = binary_content.find(b"<AppendedData")
    data_start = binary_content.find(b"_", appended_tag_start) + 1

    # Get header type
    header_type = root.get("header_type", "UInt32")
    header_size = {"UInt32": 4, "UInt64": 8}.get(header_type, 4)

    coords_list: list[np.ndarray] = []
    point_fields: dict[str, np.ndarray] = {}
    cell_fields: dict[str, np.ndarray] = {}

    dtype_map = {
        "Float32": np.float32,
        "Float64": np.float64,
        "Int32": np.int32,
        "Int64": np.int64,
    }

    def _read_array(data_array: ET.Element, offset: int) -> np.ndarray:
        """Read a single DataArray from appended binary data."""
        data_type = data_array.get("type", "Float32")
        n_components_str = data_array.get("NumberOfComponents")
        n_components = int(n_components_str) if n_components_str else 1

        header_offset = data_start + offset
        n_bytes = struct.unpack(
            "<I" if header_size == 4 else "<Q",
            binary_content[header_offset:header_offset + header_size],
        )[0]

        raw_start = header_offset + header_size
        dtype = dtype_map.get(data_type, np.float32)
        raw = np.frombuffer(
            binary_content[raw_start:raw_start + n_bytes], dtype=dtype,
        )
        if n_components > 1 and len(raw) % n_components == 0:
            raw = raw.reshape(-1, n_components)
        return raw

    for piece in root.iter("Piece"):
        # Coordinates
        points_elem = piece.find("Points")
        if points_elem is not None:
            data_array = points_elem.find("DataArray")
            if data_array is not None:
                offset = int(data_array.get("offset", 0))
                coords = _read_array(data_array, offset)
                coords_list.append(coords)

        # Point data
        point_data = piece.find("PointData")
        if point_data is not None:
            for data_array in point_data.findall("DataArray"):
                name = data_array.get("Name")
                if name:
                    offset = int(data_array.get("offset", 0))
                    point_fields[name] = _read_array(data_array, offset)

        # Cell data
        cell_data_elem = piece.find("CellData")
        if cell_data_elem is not None:
            for data_array in cell_data_elem.findall("DataArray"):
                name = data_array.get("Name")
                if name:
                    offset = int(data_array.get("offset", 0))
                    cell_fields[name] = _read_array(data_array, offset)

    if not coords_list:
        raise ValueError(f"No coordinates found in {vtu_path}")

    return VTUData(
        coordinates=np.vstack(coords_list),
        point_data=point_fields,
        cell_data=cell_fields,
    )


def extract_stagnation_values(data: VTUData) -> dict[str, float]:
    """Find the stagnation point (max pressure) and return flow properties.

    Args:
        data: Parsed VTU data.

    Returns:
        Dictionary with Temperature, Pressure, Density, Mach at the
        stagnation point. Missing fields are omitted.
    """
    if data.pressure is None:
        return {}

    # Stagnation point = node with maximum static pressure
    idx = int(np.argmax(data.pressure))
    result: dict[str, float] = {}

    result["Pressure"] = float(data.pressure[idx])

    if data.temperature is not None:
        result["Temperature"] = float(data.temperature[idx])
    if data.density is not None:
        result["Density"] = float(data.density[idx])
    if data.mach is not None:
        result["Mach"] = float(data.mach[idx])

    # Store stagnation coordinates
    result["x"] = float(data.coordinates[idx, 0])
    result["r"] = float(data.coordinates[idx, 1]) if data.coordinates.shape[1] > 1 else 0.0

    return result


def extract_surface_heat_flux(
    data: VTUData,
    wall_marker: str = "body",
) -> np.ndarray | None:
    """Extract heat flux along the wall surface.

    SU2 v8 writes wall heat flux in surface flow files (not in volume VTU).
    If a Heat_Flux or HeatFlux field exists in the VTU point data, extract it.
    Otherwise return None (heat flux requires MARKER_ISOTHERMAL surface output).

    Args:
        data: Parsed VTU data.
        wall_marker: Wall boundary marker name (unused; reserved for future
            surface VTU parsing).

    Returns:
        Heat flux array along the wall, or None if not available.
    """
    # Try common field names SU2 uses for wall heat flux
    for key in ("Heat_Flux", "HeatFlux", "Wall_Heat_Flux", "HeatFluxDensity"):
        if key in data.point_data:
            return data.point_data[key]
    return None
