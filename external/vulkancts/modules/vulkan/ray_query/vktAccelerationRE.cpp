
#include "vkDefs.hpp"
#include "deClock.h"
#include "vktTestCase.hpp"
#include "vktTestGroupUtil.hpp"
#include "vkCmdUtil.hpp"
#include "vkObjUtil.hpp"
#include "vkBuilderUtil.hpp"
#include "vkBarrierUtil.hpp"
#include "vkBufferWithMemory.hpp"
#include "vkImageWithMemory.hpp"
#include "vkTypeUtil.hpp"
#include "vkImageUtil.hpp"
#include "vkRayTracingUtil.hpp"
#include "deRandom.hpp"
#include "tcuTexture.hpp"
#include "tcuTextureUtil.hpp"
#include "tcuTestLog.hpp"
#include "tcuImageCompare.hpp"
#include "tcuFloat.hpp"

namespace vkt
{
namespace RayQuery
{
namespace
{
using namespace vk;
using namespace vkt;

class ReverseTestCase : public TestCase
{
public:
							ReverseTestCase		(tcu::TestContext& context, const char* name, const char* desc);
							~ReverseTestCase	(void);

	virtual void			checkSupport		(Context& context) const;
	virtual	void			initPrograms		(SourceCollections& programCollection) const;
	virtual TestInstance*	createInstance		(Context& context) const;
};

class ReverseTestInstance : public TestInstance
{
public:
									ReverseTestInstance		(Context& context);
									~ReverseTestInstance	(void);
	tcu::TestStatus					iterate							(void);
};

ReverseTestCase::ReverseTestCase (tcu::TestContext& context, const char* name, const char* desc)
	: vkt::TestCase	(context, name, desc)
{
}

ReverseTestCase::~ReverseTestCase (void)
{
}

void ReverseTestCase::checkSupport (Context& context) const
{
	context.requireDeviceFunctionality("VK_KHR_acceleration_structure");
}

void ReverseTestCase::initPrograms (SourceCollections& programCollection) const
{
	//const vk::ShaderBuildOptions	buildOptions(programCollection.usedVulkanVersion, vk::SPIRV_VERSION_1_4, 0u, true);
}

TestInstance* ReverseTestCase::createInstance (Context& context) const
{
	return new ReverseTestInstance(context);
}

ReverseTestInstance::ReverseTestInstance (Context& context)
	: vkt::TestInstance		(context)
{
}

ReverseTestInstance::~ReverseTestInstance (void)
{
}

typedef std::map<VkDeviceAddress, void *, std::greater<VkDeviceAddress>> addrmap_t;

void *mapHostPtr(const addrmap_t& map, VkDeviceAddress addr)
{
	auto iter = map.lower_bound(addr);
	VkDeviceAddress base = iter->first;
	void *hostPtr = iter->second;
	return static_cast<void *>(static_cast<uint8_t *>(hostPtr) + (addr - base));
}

void indent(unsigned level, std::stringstream& log)
{
	for (unsigned i = 0; i < level; i++)
		log << "\t";
}

void hexDump(void *data, VkDeviceAddress devAddr, unsigned dwords, unsigned level, std::stringstream& log)
{
	deUint32 *data32 = (deUint32 *)data;
	for (unsigned i = 0; i < dwords; i += 8)
	{
		indent(level, log);

		log << tcu::toHex(devAddr + i * 4) << ": ";
		for (unsigned j = 0; j < 8 && i + j < dwords; j++)
		{
			log << tcu::toHex(data32[i + j]) << " ";
		}
		log << "\n";
	}
}

uint32_t readBits(const uint32_t *data, unsigned offset, unsigned bits)
{
	data += offset / 32;
	offset = offset % 32;

	uint32_t mask = bits == 32 ? ~0u : ((1u << bits) - 1);
	if (offset + bits > 32)
		return ((data[1] << (32 - offset)) | (data[0] >> offset)) & mask;
	else
		return (data[0] >> offset) & mask;
}

int32_t readBitsSigned(const uint32_t *data, unsigned offset, unsigned bits)
{
	data += offset / 32;
	offset = offset % 32;

	int32_t result;
	if (offset + bits > 32)
		result = (data[1] << (32 - offset)) | (data[0] >> offset);
	else
		result = data[0] >> offset;
	return bits == 0 ? 0 : ((result << (32 - bits)) >> (32 - bits));
}


float castUintToFloat(uint32_t i)
{
	union { uint32_t i; float f; } u;
	u.i = i;
	return u.f;
}

uint32_t castFloatToUint(float f)
{
	union { uint32_t i; float f; } u;
	u.f = f;
	return u.i;
}

float offset(float f, uint32_t off)
{
	return castUintToFloat(castFloatToUint(f) + off);
}

void dumpFloat(uint32_t data, std::stringstream& log)
{
	log << castUintToFloat(data) << " (" << tcu::toHex(data) << ")";
}

void dumpPrimitive(const uint32_t *data, uint32_t id, unsigned typeFlags, unsigned level, std::stringstream& log)
{
	indent(level, log);
	unsigned vertices = 3;
	if (typeFlags & 1)
	{
		uint32_t instanceData = data[2 * 3];
		log << "instance " << id << " cull mask " << ((instanceData >> 8) & 0xff);
		// TODO are these actually used by anything or are these just here to match vk flags layout?
		if (instanceData & 0x1)
			log << " (facing cull disable)";
		if (instanceData & 0x2)
			log << " (flip triangle facing)";

		// matches vk force flags layout, although it can also be derived from
		// looking at the geometries in the BLAS if there are no force flags
		if (instanceData & 0x4)
			log << " (all opaque)";
		if (instanceData & 0x8)
			log << " (all non-opaque)";
		if (instanceData & 0x40)
			log << " AABBs";
		else
			log << " triangles";

		// this is probably reserved for micromap flags
		if (instanceData & 0x30)
			log << " unk " << (instanceData & 0x30);
		vertices = 2;
	}
	else
	{
		if (typeFlags & 8)
		{
			vertices = 2;
			log << "AABB ";
		}
		else
			log << "triangle ";
		log << id;
		if (typeFlags & 4)
			log << " non-opaque";
		else
			log << " opaque";
	}
	log << "\n";

	for (unsigned vtx = 0; vtx < vertices; vtx++)
	{
		indent(level, log);
		log << "(";
		for (unsigned coord = 0; coord < 3; coord++)
		{
			if (coord != 0)
				log << ", ";
			dumpFloat(data[vtx * 3 + coord], log);
		}
		log << ")\n";
	}
}

void dumpNonCompressedPrimitive(const uint32_t *data, unsigned level, std::stringstream& log)
{
	dumpPrimitive(&data[1], data[0], data[15] >> 24, level, log);
}

struct CompressedConfig
{
	unsigned prims;
	unsigned zDeltaEnd;
	uint32_t deltaSizes[4];
	uint32_t deltaBases[4];
	unsigned orientations[4];
	unsigned primidSizes[3];
	bool aabb;
};

static void decodeVert(uint32_t *dst, const uint32_t *src)
{
	for (unsigned i = 0; i < 3; i++)
		dst[i] = src[i] ^ ((src[i] & 0x80000000) ? 0x80000000 : ~0u);
}

void dumpCompressedPrimitives(const uint32_t *data, const CompressedConfig& config, unsigned level, std::stringstream& log)
{
	uint32_t primids[4];
	int32_t deltas[3][11];

	uint32_t deltaOffset = 4 * 32;

	primids[0] = data[0];
	for (unsigned tri = 1; tri < config.prims; tri++)
	{
		unsigned bits = config.primidSizes[tri - 1] * 8;
		primids[tri] = (bits == 0 ? 1 : readBitsSigned(data, deltaOffset, bits)) + primids[0];
		deltaOffset += bits;
	}

	for (unsigned coord = 0; coord < 2; coord++)
	{
		for (unsigned i = 0; i < config.prims * 3 - 1; i++)
		{
			unsigned sizeVal = (config.deltaSizes[coord] >> (2 * i)) & 0x3;
			unsigned bits = 0;
			if (sizeVal == 1)
				bits = 16;
			else if (sizeVal == 2)
				bits = 20;
			else if (sizeVal == 3)
				bits = 24;
			deltas[coord][i] = readBitsSigned(data, deltaOffset, bits);
			deltaOffset += bits;
		}
	}

	uint8_t unkBits = readBits(data, deltaOffset, 8);

	deltaOffset = config.zDeltaEnd;
	for (int i = config.prims * 3 - 2; i >= 0; i--)
	{
		unsigned sizeVal = (config.deltaSizes[2] >> (2 * i)) & 0x3;
		unsigned bits = 0;
		if (sizeVal == 1)
			bits = 16;
		else if (sizeVal == 2)
			bits = 20;
		else if (sizeVal == 3)
			bits = 24;
		deltaOffset -= bits;
		deltas[2][i] = readBitsSigned(data, deltaOffset, bits);
	}

	uint32_t vertices[12][3];

	for (unsigned coord = 0; coord < 3; coord++)
	{
		uint32_t deltaBases = config.deltaBases[coord];
		uint32_t base = data[1 + coord];
		vertices[0][coord] = base;

#if 0
		indent(level, log);
		log << "coord " << "xyz"[coord] << ", base = ";
		dumpEncodedFloat(base, log);
		log << "\n";
#endif

		for (unsigned i = 0; i < config.prims * 3 - 1; i++)
		{
			int32_t delta = deltas[coord][i];
#if 0
			indent(level + 1, log);
			log << "delta = " << (delta < 0 ? "-" : "") << tcu::toHex(delta < 0 ? -delta : delta) << "\n";
#endif

			switch (i)
			{
			case 0:
				vertices[1][coord] = vertices[0][coord] + delta;
				break;
			case 1:
				vertices[2][coord] = ((deltaBases & 0x1) ? vertices[1][coord] : vertices[0][coord]) + delta;
				break;
			case 2:
				vertices[3][coord] = ((deltaBases & 0x2) ? vertices[0][coord] : vertices[1][coord]) + delta;
				break;
			default:
				vertices[i + 1][coord] = (((deltaBases >> (i - 1)) & 1) ? vertices[i - 3][coord] : vertices[i - 1][coord]) + delta;
				break;
			}
		}
	}

	for (unsigned prim = 0; prim < config.prims; prim++)
	{
		uint32_t triVertices[9];
		switch (config.orientations[prim])
		{
		case 0:
			decodeVert(&triVertices[0], vertices[3 * prim + 0]);
			decodeVert(&triVertices[3], vertices[3 * prim + 1]);
			decodeVert(&triVertices[6], vertices[3 * prim + 2]);
			break;
		case 1:
			decodeVert(&triVertices[0], vertices[3 * prim + 2]);
			decodeVert(&triVertices[3], vertices[3 * prim + 0]);
			decodeVert(&triVertices[6], vertices[3 * prim + 1]);
			break;
		case 2:
			decodeVert(&triVertices[0], vertices[3 * prim + 1]);
			decodeVert(&triVertices[3], vertices[3 * prim + 2]);
			decodeVert(&triVertices[6], vertices[3 * prim + 0]);
			break;
		case 3:
			decodeVert(&triVertices[0], vertices[3 * prim + 0]);
			decodeVert(&triVertices[3], vertices[3 * prim + 2]);
			decodeVert(&triVertices[6], vertices[3 * prim + 1]);
			break;
		}

		dumpPrimitive(triVertices, primids[prim], (data[15] >> 24) & 0xf, level, log);
	}

#if 0
	indent(level, log);
	log << "unk bits " << tcu::toHex(unkBits) << "\n";
#endif
}

void dumpCompressedFourPrimitives(const uint32_t *data, unsigned level, std::stringstream& log)
{
	CompressedConfig config;
	config.prims = 4;
	config.zDeltaEnd = 12 * 32 + 8;

	for (unsigned coord = 0; coord < 3; coord++)
	{
		config.deltaSizes[coord] = readBits(&data[12], 22 + coord * 22, 22);
		config.deltaBases[coord] = readBits(&data[14], 24 + coord * 10, 10);
	}

	config.orientations[0] = readBits(&data[12], 8, 2);
	config.orientations[1] = readBits(&data[12], 10, 2);
	config.orientations[2] = readBits(&data[12], 12, 2);
	config.orientations[3] = readBits(&data[12], 14, 2);

	config.primidSizes[0] = readBits(&data[12], 16, 2);
	config.primidSizes[1] = readBits(&data[12], 18, 2);
	config.primidSizes[2] = readBits(&data[12], 20, 2);

	dumpCompressedPrimitives(data, config, level, log);
}

void dumpCompressedThreePrimitives(const uint32_t *data, unsigned level, std::stringstream& log)
{
	CompressedConfig config;
	config.prims = 3;
	config.zDeltaEnd = 13 * 32 + 8;

	for (unsigned coord = 0; coord < 3; coord++)
	{
		config.deltaSizes[coord] = readBits(&data[13], 12 + coord * 16, 16);
		config.deltaBases[coord] = readBits(&data[15], coord * 7, 7);
	}

	config.orientations[0] = readBits(&data[14], 28, 2);
	config.orientations[1] = readBits(&data[14], 30, 2);
	config.orientations[2] = readBits(&data[15], 22, 2);

	config.primidSizes[0] = readBits(&data[13], 8, 2);
	config.primidSizes[1] = readBits(&data[13], 10, 2);

	dumpCompressedPrimitives(data, config, level, log);

	uint32_t unkBits2 = readBits(&data[15], 21, 1);

	indent(level, log);
	log << "unk bits 2 " << tcu::toHex(unkBits2) << "\n";
}

void dumpCompressedTwoPrimitives(const uint32_t *data, unsigned level, std::stringstream& log)
{
	CompressedConfig config;
	config.prims = 2;
	config.zDeltaEnd = 14 * 32 + 8;

	for (unsigned coord = 0; coord < 3; coord++)
	{
		config.deltaSizes[coord] = readBits(&data[14], 10 + 10 * coord, 10);
		config.deltaBases[coord] = readBits(&data[15], 8 + 4 * coord, 4);
	}

	config.orientations[0] = readBits(&data[15], 20, 2);
	config.orientations[1] = readBits(&data[15], 22, 2);

	config.primidSizes[0] = readBits(&data[14], 8, 2);

	dumpCompressedPrimitives(data, config, level, log);
}

unsigned dumpInternal(const uint32_t *data, unsigned level, std::stringstream& log)
{
	unsigned numChildren = readBits(&data[15], 8, 3);

	float base[3];
	int exponents[3];
	unsigned offset = 32;
	for (unsigned i = 0; i < 3; i++)
	{
		base[i] = castUintToFloat(readBits(data, offset, 16) << 16);
		exponents[i] = (int)readBits(&data[14], 16 + 8 * i, 8) - 134;
		offset += 16;
	}

	for (unsigned child = 0; child < numChildren; child++)
	{
		indent(level, log);
		log << "child node " << child + data[0] << "\n";
		for (unsigned vert = 0; vert < 2; vert++)
		{
			indent(level, log);
			log << "(";
			for (unsigned coord = 0; coord < 3; coord++)
			{
				unsigned mantissa = readBits(data, offset, 8);
				offset += 8;

				if (coord != 0)
					log << ", ";
				log << base[coord] + ldexpf((float)mantissa, exponents[coord]);
			}
			log << ")\n";
		}
	}

	return numChildren + data[0];
}

void dumpNodes(const addrmap_t& map, VkDeviceAddress addr, unsigned level, std::stringstream& log)
{
	unsigned numNodes = 1;
	for (unsigned i = 0; i < numNodes; i++)
	{
		indent(level, log);
		log << "node " << i << "\n";

		void *nodes = mapHostPtr(map, addr);
		hexDump(nodes, addr, 16, level, log);
		uint32_t *nodes32 = (uint32_t *)nodes;

		unsigned type = (nodes32[15] >> 24) & 0xf;
		if (type & 0x2)
		{
			unsigned compressedPrims = nodes32[15] >> 28;
			if (compressedPrims == 0)
				dumpNonCompressedPrimitive(nodes32, level, log);
			else if (compressedPrims == 1)
				dumpCompressedTwoPrimitives(nodes32, level, log);
			else if (compressedPrims == 2)
				dumpCompressedThreePrimitives(nodes32, level, log);
			else if (compressedPrims == 3)
				dumpCompressedFourPrimitives(nodes32, level, log);
		}
		else
		{
			unsigned newNumNodes = dumpInternal(nodes32, level, log);
			numNodes = newNumNodes > numNodes ? newNumNodes : numNodes;
		}

		addr += 16 * 4;
	}
}

void dumpAS(const addrmap_t& map, VkDeviceAddress addr, unsigned instanceCount, std::stringstream& log)
{
	void *host = mapHostPtr(map, addr);
	deUint32 *host32 = (deUint32 *)host;

	hexDump(host, addr, 32, 0, log);

	VkDeviceAddress instancesAddr = host32[4] | ((uint64_t)host32[5] << 32);
	VkDeviceAddress nodesAddr = host32[6] | ((uint64_t)host32[7] << 32);

	dumpNodes(map, nodesAddr, 1, log);

	for (unsigned i = 0; i < instanceCount; i++)
	{
		void *instance = mapHostPtr(map, instancesAddr + 32 * 4 * i);
		uint32_t *instance32 = (uint32_t *)instance;

		indent(2, log);
		log << "instance " << i << "\n";
		hexDump(instance, instancesAddr + 32 * 4 * i, 32, 2, log);

		VkDeviceAddress blasNodesAddr = instance32[24] | ((uint64_t)instance32[25] << 32);
		dumpNodes(map, blasNodesAddr, 3, log);
	}
}

tcu::TestStatus ReverseTestInstance::iterate(void)
{
	tcu::TestLog&	log = m_context.getTestContext().getLog();
	const DeviceInterface&				vkd									= m_context.getDeviceInterface();
	const VkDevice						device								= m_context.getDevice();
	const VkQueue						queue								= m_context.getUniversalQueue();
	Allocator&							allocator							= m_context.getDefaultAllocator();
	const deUint32						queueFamilyIndex					= m_context.getUniversalQueueFamilyIndex();
	const Move<VkCommandPool>			cmdPool								= createCommandPool(vkd, device, 0, queueFamilyIndex);
	const Move<VkCommandBuffer>			cmdBuffer							= allocateCommandBuffer(vkd, device, *cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	addrmap_t addrMap;

	std::vector<de::SharedPtr<BottomLevelAccelerationStructure>>	bottomLevelAccelerationStructures;
	de::MovePtr<TopLevelAccelerationStructure>						topLevelAccelerationStructure = makeTopLevelAccelerationStructure();

	beginCommandBuffer(vkd, *cmdBuffer, 0u);


	de::MovePtr<BottomLevelAccelerationStructure>	bottomLevelAccelerationStructure = makeBottomLevelAccelerationStructure();
	bottomLevelAccelerationStructure->setGeometryCount(1u);

	tcu::Vec3 v0(2.0, 2.0, 0.0);
	tcu::Vec3 v1(3.0, 2.0, 0.0);
	tcu::Vec3 v2(2.0, 3.0, 0.0);
	tcu::Vec3 v3(3.0, 3.0, 0.0);
#if 0
	tcu::Vec3 v0(offset(0.0, 0 << 12), offset(0.0, (0 << 12)), offset(0.0, (0 << 12)));
	tcu::Vec3 v1(offset(0.0, 0 << 16), offset(1.0, -(0 << 16)), offset(0.0, (0 << 16) + 0));
	tcu::Vec3 v2(offset(0.0, 0 << 16), offset(0.0, 0 << 16), offset(1.0, 0 << 16));
	tcu::Vec3 v3(offset(1.0, 0 << 16), offset(1.0, 0 << 16), offset(1.0, 0 << 16));
#endif
#if 0
	tcu::Vec3 v3(offset(11.0, 0 << 16), offset(13.0, (0 << 16) + 0), offset(15.0, (0 << 16) + 0));
	tcu::Vec3 v4(offset(20.0, 0 << 16), offset(22.0, 0 << 16), offset(24.0, 0 << 16));
	tcu::Vec3 v5(offset(21.0, 0 << 16), offset(23.0, (0 << 16) + 0), offset(25.0, (0 << 16) + 0));
	tcu::Vec3 v0(offset(1.0, 0 << 12), 0., 0.);
	tcu::Vec3 v1(offset(1.0, -1 << 12), 0., 0.);
	tcu::Vec3 v2(offset(1.0, 2 << 12), 0., 0.);
	tcu::Vec3 v3(offset(1.0, (3 << 12) + 0), 0., 0.);
	tcu::Vec3 v4(offset(1.0, (4 << 12) + 0), 0., 0.);
	tcu::Vec3 v5(offset(1.0, (5 << 12) + 0), 0., 0.);
	tcu::Vec3 v6(offset(1.0, 6 << 12), 0., 0.);
	tcu::Vec3 v7(offset(1.0, (7 << 12) + 0), 0., 0.);
	tcu::Vec3 v8(offset(1.0, (8 << 12) + 0), 0., 0.);
	tcu::Vec3 v9(offset(1.0, (9 << 12) + 0), 0., 0.);
	tcu::Vec3 v10(offset(1.0, (10 << 12) + 0), 0., 0.);
	tcu::Vec3 v11(offset(1.0, (11 << 12) + 0), 0., 0.);
#endif

	bottomLevelAccelerationStructure->setGeometryCount(1);

	unsigned count = 0x7f;
	de::SharedPtr<RaytracedGeometryBase> geometry;
	geometry = makeRaytracedGeometry(VK_GEOMETRY_TYPE_TRIANGLES_KHR, VK_FORMAT_R32G32B32_SFLOAT, VK_INDEX_TYPE_UINT32, false);
	geometry->setGeometryFlags(0u);
	geometry->addVertex(v0);
	geometry->addVertex(v1);
	geometry->addVertex(v2);
	geometry->addVertex(v3);
#if 0
	geometry->addVertex(v3);
	geometry->addVertex(v4);
	geometry->addVertex(v5);
	geometry->addVertex(v4);
	geometry->addVertex(v5);
	geometry->addVertex(v6);
	geometry->addVertex(v7);
	geometry->addVertex(v8);
	geometry->addVertex(v9);
	geometry->addVertex(v10);
	geometry->addVertex(v11);
#endif

#if 1
	geometry->addIndex(0);
	geometry->addIndex(1);
	geometry->addIndex(2);
	geometry->addIndex(1);
	geometry->addIndex(2);
	geometry->addIndex(3);
#endif

	de::SharedPtr<RaytracedGeometryBase> geometry2;
	geometry2 = makeRaytracedGeometry(VK_GEOMETRY_TYPE_AABBS_KHR, VK_FORMAT_R32G32B32_SFLOAT, VK_INDEX_TYPE_NONE_KHR, false);
	geometry2->setGeometryFlags(0u);
	geometry2->addVertex(v0);
	geometry2->addVertex(v3);

	bottomLevelAccelerationStructure->addGeometry(geometry);
	//bottomLevelAccelerationStructure->addGeometry(geometry2);
	bottomLevelAccelerationStructures.push_back(de::SharedPtr<BottomLevelAccelerationStructure>(bottomLevelAccelerationStructure.release()));

	for (auto& blas : bottomLevelAccelerationStructures)
	{
		blas->setBuildType						(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR);
		blas->setBuildFlags						(0);
		blas->setUseArrayOfPointers				(false);
		blas->setCreateGeneric					(false);
		blas->setBuildWithoutGeometries			(false);
		blas->setBuildWithoutPrimitives			(false);
		blas->createAndBuild					(vkd, device, *cmdBuffer, allocator);
		auto& allocation = blas->getAllocation();
		auto& buffer = blas->getBuffer();
		VkDeviceAddress baseAddr = getBufferDeviceAddress(vkd, device, buffer, 0);
		addrMap[baseAddr] = allocation.getHostPtr();
	}

	const VkTransformMatrixKHR			transformMatrixKHR =
	{
		{								//  float	matrix[3][4];
			{ 1.0f, 0.0f, 0.0f, 0.0f },
			{ 0.0f, 1.0f, 0.0f, 0.0f },
			{ 0.0f, 0.0f, 1.0f, 0.0f },
		}
	};
	topLevelAccelerationStructure->addInstance(bottomLevelAccelerationStructures[0], transformMatrixKHR, 0u, 0xFFu, 0u, 0u);

	topLevelAccelerationStructure->setBuildType					(VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR);
	topLevelAccelerationStructure->setBuildFlags				(0);
	topLevelAccelerationStructure->setBuildWithoutPrimitives	(false);
	topLevelAccelerationStructure->setUseArrayOfPointers		(false);
	topLevelAccelerationStructure->setCreateGeneric				(false);
	topLevelAccelerationStructure->setInactiveInstances			(false);
	topLevelAccelerationStructure->createAndBuild				(vkd, device, *cmdBuffer, allocator);

	endCommandBuffer(vkd, *cmdBuffer);
	submitCommandsAndWait(vkd, device, queue, cmdBuffer.get());

	const VkAccelerationStructureDeviceAddressInfoKHR addressInfo =
	{
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,	// VkStructureType				sType
		DE_NULL,															// const void*					pNext
		*topLevelAccelerationStructure->getPtr()							// VkAccelerationStructureKHR	accelerationStructure
	};

	VkDeviceAddress vkda = vkd.getAccelerationStructureDeviceAddressKHR(device, &addressInfo);
	auto& allocation = topLevelAccelerationStructure->getAllocation();
	auto& buffer = topLevelAccelerationStructure->getBuffer();
	VkDeviceAddress baseAddr = getBufferDeviceAddress(vkd, device, buffer, 0);
	addrMap[baseAddr] = allocation.getHostPtr();

	if (baseAddr > vkda)
	{
		return tcu::TestStatus::fail("Buffer out of bounds");
	}

	log << tcu::TestLog::Message << "offset " << vkda - baseAddr << tcu::TestLog::EndMessage;
	std::stringstream ss;
	ss << "\n";
	dumpAS(addrMap, vkda, 1, ss);
	log << tcu::TestLog::Message << ss.str() << tcu::TestLog::EndMessage;

	return tcu::TestStatus::pass("Pass");
}
}	// anonymous

tcu::TestCaseGroup*	createRETests(tcu::TestContext& testCtx)
{
	de::MovePtr<tcu::TestCaseGroup> group(new tcu::TestCaseGroup(testCtx, "re", "reverse engineering"));
	group->addChild(new ReverseTestCase(group->getTestContext(), "re", ""));

	return group.release();
}

}	// RayQuery

}	// vkt
