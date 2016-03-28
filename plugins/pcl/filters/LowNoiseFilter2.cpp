/******************************************************************************
* Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/

#include "LowNoiseFilter2.hpp"

#include <algorithm>

#include "PCLConversions.hpp"

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/pdal_macros.hpp>

#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/morphological_filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.lownoise2", "Statistical outlier removal",
               "http://pdal.io/stages/filters.lownoise2.html");

CREATE_SHARED_PLUGIN(1, 0, LowNoiseFilter2, Filter, s_info)

std::string LowNoiseFilter2::getName() const
{
    return s_info.name;
}

Options LowNoiseFilter2::getDefaultOptions()
{
    Options options;
    options.add("multiplier", 2, "Standard deviation threshold");
    options.add("resolution", 10, "Resolution");
    options.add("classify", true, "Apply classification labels?");
    options.add("extract", false, "Extract ground returns?");
    return options;
}

void LowNoiseFilter2::processOptions(const Options& options)
{
    m_multiplier = options.getValueOrDefault<double>("multiplier", 2);
    m_resolution = options.getValueOrDefault<double>("resolution", 10);
    m_classify = options.getValueOrDefault<bool>("classify", true);
    m_extract = options.getValueOrDefault<bool>("extract", false);
}

void LowNoiseFilter2::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Dimension::Id::Classification);
}

PointViewSet LowNoiseFilter2::run(PointViewPtr input)
{
    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);
    log()->get(LogLevel::Debug2) << "Process LowNoiseFilter2...\n";

    // convert PointView to PointXYZ
    typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
    Cloud::Ptr cloud(new Cloud);
    BOX3D bounds;
    input->calculateBounds(bounds);
    pclsupport::PDALtoPCD(input, *cloud, bounds);

    // PCL should provide console output at similar verbosity level as PDAL
    int level = log()->getLevel();
    switch (level)
    {
        case 0:
            pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
            break;
        case 1:
            pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
            break;
        case 2:
            pcl::console::setVerbosityLevel(pcl::console::L_WARN);
            break;
        case 3:
            pcl::console::setVerbosityLevel(pcl::console::L_INFO);
            break;
        case 4:
            pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
            break;
        default:
            pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
            break;
    }

    // Compute global mean, variance, and standard deviation for Z dimension
    double sum = 0.0, sumsqr = 0.0;
    for (auto const& p : cloud->points)
    {
      sum += p.z;
      sumsqr += p.z * p.z;
    }

    double mean = sum / cloud->size();
    double variance = (sumsqr - sum * sum / cloud->size()) / (cloud->size() - 1);
    double stdev = std::sqrt(variance);

    // Compute morphologically closed copy of the cloud.
    Cloud::Ptr cloud_f(new Cloud);
    pcl::applyMorphologicalOperator<pcl::PointXYZ> (cloud, m_resolution, pcl::MORPH_CLOSE, *cloud_f);

    // Compute mean, variance, and standard deviation for change in the Z
    // dimension from the morphed cloud to the original cloud.
    std::vector<float> delta_z(cloud_f->points.size ());
    for (size_t i = 0; i < cloud_f->points.size(); ++i)
    {
        delta_z[i] = cloud_f->points[i].z - cloud->points[i].z;
    }

    double sum_dz = 0.0, sumsqr_dz = 0.0;
    for (auto const& delta : delta_z)
    {
      sum_dz += delta;
      sumsqr_dz += (double) delta * delta;
    }

    double mean_dz = sum_dz / delta_z.size();
    double variance_dz = (sumsqr_dz - sum_dz * sum_dz / delta_z.size()) / (delta_z.size() - 1);
    double stdev_dz = std::sqrt(variance_dz);


    // Designate inliers and outliers based on delta-Z between original point
    // cloud and the morphologically closed version.
    std::vector<int> inliers, outliers;
    inliers.reserve (cloud->size ());
    outliers.reserve (cloud->size () / 2);

    //we're looking for points that are below ground (so they should be
    //well below the mean Z of the cloud) and move UP when morphed closed
    //(so they're delta_z should be well above the mean delta_Z)
    float z_threshold = mean - m_multiplier * stdev;
    float delta_z_threshold = mean_dz + stdev_dz;

    for (int i = 0; i < (int) cloud->size(); ++i)
    {
        //float diff = cloud->points[i].z - cloud_f->points[i].z;
        //if (diff < -100)
        //    outliers.push_back(i);
        //else
        //    inliers.push_back(i);

        const float& z = cloud->points[i].z;
        const float& dz = delta_z[i];
        if (z < z_threshold && dz > delta_z_threshold)
            outliers.push_back(i);
        else
            inliers.push_back(i);
    }


    log()->get(LogLevel::Debug2) << inliers.size() << " points detected above threshold" << std::endl;
    log()->get(LogLevel::Debug2) << outliers.size() << " points removed by morph" << std::endl;


    PointViewSet viewSet;
    if (inliers.empty())
    {
        log()->get(LogLevel::Warning) << "Requested filter would remove all points. Try increasing the multiplier.\n";
        viewSet.insert(input);
        return viewSet;
    }

    if (!outliers.empty() && (m_classify || m_extract))
    {

        if (m_classify)
        {
            log()->get(LogLevel::Debug2) << "Labeled " << outliers.size() << " outliers as noise!\n";

            // set the classification label of outlier returns as 7
            // (corresponding to ASPRS LAS specification for low noise)
            for (const auto& i : outliers)
            {
                input->setField(Dimension::Id::Classification, i, 7);
            }

            viewSet.insert(input);
        }

        if (m_extract)
        {
            log()->get(LogLevel::Debug2) << "Extracted " << inliers.size() << " inliers!\n";

            // create new PointView containing only outliers
            PointViewPtr output = input->makeNew();
            log()->get(LogLevel::Debug) << "made\n";
            for (const auto& i : inliers)
            {
                output->appendPoint(*input, i);
            }
            log()->get(LogLevel::Debug) << "appended\n";

            viewSet.erase(input);
            log()->get(LogLevel::Debug) << "erased\n";
            viewSet.insert(output);
            log()->get(LogLevel::Debug) << "inserted\n";
        }
    }
    else
    {
        if (outliers.empty())
            log()->get(LogLevel::Warning) << "Filtered cloud has no outliers!\n";

        if (!(m_classify || m_extract))
            log()->get(LogLevel::Warning) << "Must choose --classify or --extract\n";

        // return the input buffer unchanged
        viewSet.insert(input);
    }

    return viewSet;
}

} // namespace pdal
